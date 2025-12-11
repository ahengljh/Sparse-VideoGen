"""
Dynamic Layer Offloading for Video Diffusion Models

This module provides intelligent CPU-GPU memory management for running large
video diffusion models (like HunyuanVideo) on consumer GPUs with limited VRAM.

Key features:
1. Layer-wise offloading: Only one transformer layer on GPU at a time
2. Async prefetching: Use CUDA streams to hide CPU-GPU transfer latency
3. Pinned memory: Faster transfers with page-locked CPU memory
4. Smart scheduling: Prefetch next layer while computing current layer

This enables running HunyuanVideo (~13GB) on 24GB GPUs like RTX 4090.
"""

import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
from contextlib import contextmanager
import threading

import torch
import torch.nn as nn

from .logger import logger
from .timer import time_logging_decorator


@dataclass
class OffloadConfig:
    """Configuration for dynamic layer offloading."""

    # Device settings
    compute_device: str = "cuda"
    offload_device: str = "cpu"

    # Memory settings
    use_pinned_memory: bool = True  # Use page-locked memory for faster transfers

    # Layer batching - how many layers to keep on GPU simultaneously
    # Higher = faster but more memory, Lower = slower but less memory
    # Recommended: 4-8 for 24GB GPU, 10-15 for 40GB GPU
    num_layers_on_gpu: int = 6

    # Prefetching settings
    enable_prefetch: bool = True    # Async prefetch next layer
    prefetch_count: int = 2         # Number of layers to prefetch ahead

    # Components to keep on GPU (don't offload)
    keep_on_gpu: List[str] = None   # e.g., ["text_encoder", "vae"]

    # Memory management
    empty_cache_frequency: int = 10  # Call empty_cache every N layers

    # Memory budget (optional) - if set, auto-tune num_layers_on_gpu
    max_memory_gb: float = None     # e.g., 20.0 for 24GB GPU with headroom

    # Debugging
    verbose: bool = False


class LayerOffloadManager:
    """
    Manages dynamic CPU-GPU offloading for transformer layers.

    This class implements a sliding window approach where multiple layers
    are kept on GPU simultaneously for better throughput, while still
    fitting within memory constraints.

    Key features:
    1. Sliding window of N layers on GPU (configurable)
    2. Async prefetching of upcoming layers
    3. Automatic memory management

    Usage:
        manager = LayerOffloadManager(model, config)
        manager.prepare_for_inference()

        for layer_idx in range(num_layers):
            manager.ensure_layer_on_gpu(layer_idx)
            output = layer(input)
            manager.layer_forward_complete(layer_idx)
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: OffloadConfig,
        double_blocks_attr: str = "transformer_blocks",
        single_blocks_attr: str = "single_transformer_blocks",
    ):
        """
        Initialize the offload manager.

        Args:
            transformer: The transformer module to manage
            config: Offloading configuration
            double_blocks_attr: Attribute name for double stream blocks
            single_blocks_attr: Attribute name for single stream blocks
        """
        self.transformer = transformer
        self.config = config
        self.double_blocks_attr = double_blocks_attr
        self.single_blocks_attr = single_blocks_attr

        # Get layer lists
        self.double_blocks = getattr(transformer, double_blocks_attr, [])
        self.single_blocks = getattr(transformer, single_blocks_attr, [])
        self.all_blocks = list(self.double_blocks) + list(self.single_blocks)
        self.num_layers = len(self.all_blocks)

        # Calculate per-layer memory
        self._layer_memory_mb = self._estimate_layer_memory()

        # Auto-tune num_layers_on_gpu if max_memory_gb is specified
        if config.max_memory_gb is not None:
            self._auto_tune_layers_on_gpu()

        # Track layer locations
        self._layer_on_gpu: Dict[int, bool] = {}
        self._layer_pinned: Dict[int, bool] = {}
        self._layers_on_gpu_set: set = set()  # Track which layers are currently on GPU

        # CUDA streams for async operations
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._compute_stream: Optional[torch.cuda.Stream] = None

        # Prefetch state
        self._prefetch_in_progress: Dict[int, bool] = {}
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}

        # Current window position
        self._current_window_start: int = 0

        # Statistics
        self.stats = {
            'gpu_loads': 0,
            'gpu_offloads': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'cache_clears': 0,
            'window_slides': 0,
        }

        self._initialized = False

    def _estimate_layer_memory(self) -> float:
        """Estimate memory per layer in MB."""
        if len(self.all_blocks) == 0:
            return 0.0

        # Sample first layer
        layer = self.all_blocks[0]
        total_params = sum(p.numel() * p.element_size() for p in layer.parameters())
        total_buffers = sum(b.numel() * b.element_size() for b in layer.buffers() if b is not None)

        return (total_params + total_buffers) / (1024 * 1024)

    def _auto_tune_layers_on_gpu(self):
        """Auto-tune the number of layers to keep on GPU based on memory budget."""
        if self.config.max_memory_gb is None:
            return

        # Reserve memory for:
        # - VAE: ~300MB
        # - Activations: ~3GB (conservative estimate for 720p video)
        # - CUDA overhead: ~500MB
        reserved_gb = 4.0

        available_gb = self.config.max_memory_gb - reserved_gb
        available_mb = available_gb * 1024

        # Calculate how many layers fit
        if self._layer_memory_mb > 0:
            max_layers = int(available_mb / self._layer_memory_mb)
            # Clamp to reasonable range
            self.config.num_layers_on_gpu = max(1, min(max_layers, self.num_layers))

            logger.info(f"Auto-tuned layers on GPU: {self.config.num_layers_on_gpu} "
                       f"(~{self._layer_memory_mb:.1f}MB/layer, "
                       f"{available_gb:.1f}GB available)")

    def prepare_for_inference(self):
        """
        Prepare the model for offloaded inference.

        This sets up CUDA streams and tracking for layers.
        Layers should already be on CPU (moved by setup_offloading_for_pipeline).
        Call this before starting inference.
        """
        if self._initialized:
            return

        logger.info(f"Preparing {self.num_layers} layers for offloaded inference...")

        # Create CUDA streams
        if self.config.enable_prefetch:
            self._prefetch_stream = torch.cuda.Stream()
            self._compute_stream = torch.cuda.Stream()

        # Check layer locations and set up tracking
        # Layers should already be on CPU from setup_offloading_for_pipeline
        for idx, layer in enumerate(self.all_blocks):
            # Check if layer is on CPU or GPU
            first_param = next(layer.parameters(), None)
            if first_param is not None:
                is_on_cpu = first_param.device.type == 'cpu'
                self._layer_on_gpu[idx] = not is_on_cpu

                # If on CPU but we want pinned memory, convert to pinned
                if is_on_cpu and self.config.use_pinned_memory:
                    self._ensure_pinned_memory(idx)

        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()

        self._initialized = True

        num_on_cpu = sum(1 for v in self._layer_on_gpu.values() if not v)
        logger.info(f"Offload manager initialized. {num_on_cpu}/{self.num_layers} layers on CPU.")

    def _ensure_pinned_memory(self, layer_idx: int):
        """Convert a CPU layer to use pinned memory for faster GPU transfers."""
        layer = self.all_blocks[layer_idx]

        for param in layer.parameters():
            if param.device.type == 'cpu' and not param.data.is_pinned():
                # Create pinned tensor and copy data
                pinned_tensor = torch.empty_like(param.data, pin_memory=True)
                pinned_tensor.copy_(param.data)
                param.data = pinned_tensor

        self._layer_pinned[layer_idx] = True

    def _move_layer_to_cpu(self, layer_idx: int, use_pinned: bool = False):
        """Move a layer to CPU, optionally with pinned memory."""
        layer = self.all_blocks[layer_idx]

        if use_pinned:
            # Move to CPU with pinned memory for faster future transfers
            for param in layer.parameters():
                if param.device.type != 'cpu':
                    cpu_tensor = param.data.cpu()
                    if cpu_tensor.is_pinned():
                        param.data = cpu_tensor
                    else:
                        # Create pinned tensor
                        pinned_tensor = torch.empty_like(cpu_tensor, pin_memory=True)
                        pinned_tensor.copy_(cpu_tensor)
                        param.data = pinned_tensor

            for buffer_name, buffer in layer.named_buffers():
                if buffer is not None and buffer.device.type != 'cpu':
                    cpu_tensor = buffer.cpu()
                    if not cpu_tensor.is_pinned():
                        pinned_tensor = torch.empty_like(cpu_tensor, pin_memory=True)
                        pinned_tensor.copy_(cpu_tensor)
                        setattr(layer, buffer_name.split('.')[-1], pinned_tensor)

            self._layer_pinned[layer_idx] = True
        else:
            layer.to('cpu')
            self._layer_pinned[layer_idx] = False

        self._layer_on_gpu[layer_idx] = False

    def _move_layer_to_gpu(self, layer_idx: int, non_blocking: bool = False):
        """Move a layer to GPU."""
        layer = self.all_blocks[layer_idx]
        layer.to(self.config.compute_device, non_blocking=non_blocking)
        self._layer_on_gpu[layer_idx] = True
        self.stats['gpu_loads'] += 1

    def _offload_layer_from_gpu(self, layer_idx: int):
        """Offload a layer from GPU back to CPU."""
        if not self._layer_on_gpu.get(layer_idx, False):
            return

        self._move_layer_to_cpu(layer_idx, use_pinned=self.config.use_pinned_memory)
        self.stats['gpu_offloads'] += 1

    @time_logging_decorator("Level 4 - Layer prefetch")
    def _start_prefetch(self, layer_idx: int):
        """Start async prefetch of a layer."""
        if not self.config.enable_prefetch:
            return

        if layer_idx >= self.num_layers:
            return

        if self._layer_on_gpu.get(layer_idx, False):
            return  # Already on GPU

        if self._prefetch_in_progress.get(layer_idx, False):
            return  # Already prefetching

        self._prefetch_in_progress[layer_idx] = True

        # Create event to track completion
        event = torch.cuda.Event()
        self._prefetch_events[layer_idx] = event

        # Async transfer on prefetch stream
        with torch.cuda.stream(self._prefetch_stream):
            self._move_layer_to_gpu(layer_idx, non_blocking=True)
            event.record()

        if self.config.verbose:
            logger.debug(f"Started prefetch for layer {layer_idx}")

    def _wait_for_prefetch(self, layer_idx: int):
        """Wait for prefetch of a layer to complete."""
        if layer_idx in self._prefetch_events:
            self._prefetch_events[layer_idx].synchronize()
            del self._prefetch_events[layer_idx]
            self._prefetch_in_progress[layer_idx] = False
            self.stats['prefetch_hits'] += 1
        else:
            self.stats['prefetch_misses'] += 1

    def _evict_oldest_layers(self, keep_layer_idx: int):
        """
        Evict oldest layers to make room in the GPU sliding window.

        This implements a sliding window strategy: keep the most recent N layers
        on GPU, where N = num_layers_on_gpu.

        Args:
            keep_layer_idx: The layer we're about to use (must stay on GPU)
        """
        # Calculate the window bounds
        # Window should cover [keep_layer_idx - num_layers_on_gpu + 1, keep_layer_idx]
        window_end = keep_layer_idx
        window_start = max(0, keep_layer_idx - self.config.num_layers_on_gpu + 1)

        # Find layers outside the window
        layers_to_evict = []
        for idx in list(self._layers_on_gpu_set):
            if idx < window_start:
                layers_to_evict.append(idx)

        # Evict layers outside the window
        for idx in layers_to_evict:
            self._offload_layer_from_gpu(idx)
            self._layers_on_gpu_set.discard(idx)
            self.stats['window_slides'] += 1

            if self.config.verbose:
                logger.debug(f"Evicted layer {idx} (window: [{window_start}, {window_end}])")

    @time_logging_decorator("Level 3 - Ensure layer on GPU")
    def ensure_layer_on_gpu(self, layer_idx: int):
        """
        Ensure a layer is on GPU, loading it if necessary.

        This implements a sliding window approach:
        - Keep up to N layers on GPU simultaneously
        - Evict oldest layers when window moves forward
        - Prefetch upcoming layers

        This should be called before using a layer for computation.
        """
        if not self._initialized:
            self.prepare_for_inference()

        # Check if already on GPU
        if self._layer_on_gpu.get(layer_idx, False):
            # Wait for any in-progress prefetch
            if self._prefetch_in_progress.get(layer_idx, False):
                self._wait_for_prefetch(layer_idx)
            # Add to tracking set
            self._layers_on_gpu_set.add(layer_idx)
            return

        # Check if prefetch is in progress
        if self._prefetch_in_progress.get(layer_idx, False):
            self._wait_for_prefetch(layer_idx)
        else:
            # Synchronous load (prefetch miss)
            self._move_layer_to_gpu(layer_idx, non_blocking=False)
            self.stats['prefetch_misses'] += 1

        # Track this layer as being on GPU
        self._layers_on_gpu_set.add(layer_idx)

        # Evict old layers that fall outside the sliding window
        self._evict_oldest_layers(layer_idx)

        # Start prefetching next layers
        for i in range(1, self.config.prefetch_count + 1):
            next_idx = layer_idx + i
            if next_idx < self.num_layers:
                self._start_prefetch(next_idx)

    @time_logging_decorator("Level 3 - Layer forward complete")
    def layer_forward_complete(self, layer_idx: int):
        """
        Called after a layer's forward pass is complete.

        With sliding window approach, we don't immediately offload.
        Layers are evicted when the window slides forward in ensure_layer_on_gpu().

        We only periodically clear CUDA cache for memory efficiency.
        """
        # Note: With sliding window, we keep layers on GPU until they fall
        # outside the window. Eviction is handled by _evict_oldest_layers().

        # Periodically clear CUDA cache
        if (layer_idx + 1) % self.config.empty_cache_frequency == 0:
            torch.cuda.empty_cache()
            self.stats['cache_clears'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        total_ops = self.stats['gpu_loads'] + self.stats['prefetch_hits']
        if total_ops > 0:
            prefetch_ratio = self.stats['prefetch_hits'] / total_ops
        else:
            prefetch_ratio = 0.0

        return {
            **self.stats,
            'prefetch_hit_ratio': prefetch_ratio,
            'num_layers': self.num_layers,
        }

    def print_statistics(self):
        """Print offloading statistics."""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("Dynamic Layer Offloading Statistics")
        print("=" * 60)
        print(f"Total layers:           {stats['num_layers']}")
        print(f"Layers kept on GPU:     {self.config.num_layers_on_gpu}")
        print(f"Layer memory:           ~{self._layer_memory_mb:.1f}MB each")
        print(f"GPU loads:              {stats['gpu_loads']}")
        print(f"GPU offloads:           {stats['gpu_offloads']}")
        print(f"Window slides:          {stats['window_slides']}")
        print(f"Prefetch hits:          {stats['prefetch_hits']}")
        print(f"Prefetch misses:        {stats['prefetch_misses']}")
        print(f"Prefetch hit ratio:     {stats['prefetch_hit_ratio']*100:.1f}%")
        print(f"Cache clears:           {stats['cache_clears']}")
        print("=" * 60 + "\n")

    def reset(self):
        """Reset state for a new inference run."""
        self._prefetch_in_progress.clear()
        self._prefetch_events.clear()
        # Reset stats
        self.stats = {k: 0 for k in self.stats}


class OffloadedModuleWrapper(nn.Module):
    """
    Wrapper that intercepts forward calls to manage offloading.

    This wraps a transformer block and ensures it's on GPU during forward,
    then offloads it after completion.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        offload_manager: LayerOffloadManager,
    ):
        super().__init__()
        self._module = module
        self._layer_idx = layer_idx
        self._offload_manager = offload_manager

    def forward(self, *args, **kwargs):
        # Ensure layer is on GPU
        self._offload_manager.ensure_layer_on_gpu(self._layer_idx)

        # Forward pass
        output = self._module(*args, **kwargs)

        # Offload layer
        self._offload_manager.layer_forward_complete(self._layer_idx)

        return output

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)


class TextEncoderOffloadManager:
    """
    Manages text encoder offloading using pre-encode strategy.

    The clean approach:
    1. Keep text encoders on GPU initially
    2. Pre-encode the prompt BEFORE setting up transformer offloading
    3. Then offload text encoders to CPU
    4. Pass pre-computed embeddings to pipeline

    This avoids device mismatch issues that occur with hook-based approaches.
    """

    def __init__(
        self,
        pipe,
        compute_device: str = "cuda",
        verbose: bool = False,
    ):
        self.pipe = pipe
        self.compute_device = compute_device
        self.verbose = verbose
        self._text_encoders: Dict[str, nn.Module] = {}
        self._is_offloaded: bool = False

        # Find all text encoders
        encoder_names = ['text_encoder', 'text_encoder_2', 'text_encoder_3']
        for name in encoder_names:
            if hasattr(pipe, name):
                encoder = getattr(pipe, name)
                if encoder is not None:
                    self._text_encoders[name] = encoder

        if self.verbose:
            logger.info(f"TextEncoderOffloadManager: Found {len(self._text_encoders)} text encoder(s)")

    def get_memory_estimate(self) -> float:
        """Estimate memory used by text encoders (in GB)."""
        total_bytes = 0
        for encoder in self._text_encoders.values():
            for param in encoder.parameters():
                total_bytes += param.numel() * param.element_size()
        return total_bytes / (1024 ** 3)

    def ensure_on_gpu(self):
        """Ensure text encoders are on GPU for encoding."""
        for name, encoder in self._text_encoders.items():
            encoder.to(self.compute_device)
            if self.verbose:
                logger.info(f"{name} moved to GPU")
        self._is_offloaded = False

    def offload_to_cpu(self):
        """Offload text encoders to CPU after encoding."""
        for name, encoder in self._text_encoders.items():
            encoder.to('cpu')
            if self.verbose:
                logger.info(f"{name} offloaded to CPU")
        torch.cuda.empty_cache()
        self._is_offloaded = True

        memory_freed = self.get_memory_estimate()
        logger.info(f"Text encoders offloaded to CPU. Freed ~{memory_freed:.1f}GB GPU memory")

    def pre_encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_videos_per_prompt: int = 1,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-encode prompt while text encoders are on GPU.

        Returns dict with prompt_embeds, pooled_prompt_embeds, prompt_attention_mask
        that can be passed directly to the pipeline.
        """
        if device is None:
            device = self.compute_device
        if dtype is None:
            dtype = torch.bfloat16

        # Ensure text encoders are on GPU
        self.ensure_on_gpu()

        # Use pipeline's encode_prompt method
        # Note: HunyuanVideoPipeline.encode_prompt doesn't support negative_prompt directly
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            dtype=dtype,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # Offload text encoders to CPU
        self.offload_to_cpu()

        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
        }


def pre_encode_and_offload(
    pipe,
    prompt: str,
    prompt_2: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function: Pre-encode prompt then offload text encoders.

    This should be called BEFORE setting up transformer offloading.

    Args:
        pipe: HunyuanVideoPipeline
        prompt: The text prompt
        prompt_2: Optional secondary prompt for second encoder
        device: Target device for embeddings
        dtype: Data type for embeddings
        max_sequence_length: Maximum token sequence length

    Returns:
        Dict with prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    Usage:
        # 1. Pre-encode prompt (text encoders on GPU temporarily)
        prompt_embeds = pre_encode_and_offload(pipe, prompt)

        # 2. Set up transformer offloading
        manager, hooks = enable_offloading(pipe)

        # 3. Run pipeline with pre-computed embeddings
        output = pipe(prompt_embeds=prompt_embeds['prompt_embeds'], ...)
    """
    manager = TextEncoderOffloadManager(pipe, compute_device=device, verbose=True)
    return manager.pre_encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
    )


def setup_offloading_for_pipeline(
    pipe,
    config: Optional[OffloadConfig] = None,
    keep_vae_on_gpu: bool = True,
    offload_text_encoders: bool = True,  # If True, assumes text encoders are already on CPU (pre-encoded)
) -> LayerOffloadManager:
    """
    Set up dynamic offloading for a HunyuanVideo pipeline.

    IMPORTANT: If using offload_text_encoders=True (default), you should call
    pre_encode_and_offload() BEFORE this function to pre-compute prompt embeddings.

    This function:
    1. Keeps the transformer on CPU initially
    2. Wraps transformer blocks with offloading logic
    3. Keeps VAE on GPU for decoding

    Args:
        pipe: HunyuanVideoPipeline instance
        config: Offloading configuration
        keep_vae_on_gpu: Whether to keep VAE on GPU (needed for decode)
        offload_text_encoders: If True, ensures text encoders stay on CPU
                              (assumes pre_encode_and_offload was called first)

    Returns:
        LayerOffloadManager instance
    """
    if config is None:
        config = OffloadConfig()

    logger.info("Setting up dynamic layer offloading for HunyuanVideo...")

    # Create offload manager for transformer
    offload_manager = LayerOffloadManager(
        pipe.transformer,
        config,
        double_blocks_attr="transformer_blocks",
        single_blocks_attr="single_transformer_blocks",
    )

    # IMPORTANT: Don't move the entire transformer to CPU!
    # The transformer has components that need to stay on GPU:
    # - time_text_embed: processes timestep + pooled_projections
    # - x_embedder: processes input latents
    # - context_embedder: processes prompt_embeds
    # - norm_out, proj_out: final output processing
    #
    # Only the transformer_blocks and single_transformer_blocks should be offloaded.

    # First, move the entire transformer to GPU to ensure all components are there
    pipe.transformer.to(config.compute_device)

    # Then, move ONLY the transformer blocks to CPU (they will be loaded on-demand)
    for block in pipe.transformer.transformer_blocks:
        block.to('cpu')
    for block in pipe.transformer.single_transformer_blocks:
        block.to('cpu')

    logger.info(f"Transformer embedders/norms on GPU, {len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)} blocks on CPU")

    # VAE - keep on GPU for decoding (it's small ~300MB)
    if keep_vae_on_gpu:
        pipe.vae.to(config.compute_device)
        logger.info("VAE kept on GPU")
    else:
        pipe.vae.to('cpu')
        logger.info("VAE on CPU")

    # Text encoders - ensure they're on CPU if offloading is enabled
    # (They should already be on CPU if pre_encode_and_offload was called)
    if offload_text_encoders:
        text_encoder_names = ['text_encoder', 'text_encoder_2', 'text_encoder_3']
        for enc_name in text_encoder_names:
            if hasattr(pipe, enc_name):
                encoder = getattr(pipe, enc_name)
                if encoder is not None:
                    encoder.to('cpu')
        logger.info("Text encoders on CPU (use pre-computed embeddings)")

    # Prepare offload manager (this will set up tracking for the blocks)
    offload_manager.prepare_for_inference()

    # Clear CUDA cache after setup
    torch.cuda.empty_cache()
    gc.collect()

    # Log memory status
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after setup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return offload_manager


def create_offloaded_forward_hook(
    offload_manager: LayerOffloadManager,
    layer_idx: int,
) -> Callable:
    """
    Create a forward pre-hook that ensures the layer is on GPU.

    This is an alternative to wrapping modules - uses hooks instead.
    """
    def hook(module, args):
        offload_manager.ensure_layer_on_gpu(layer_idx)
        return args

    return hook


def create_offloaded_forward_post_hook(
    offload_manager: LayerOffloadManager,
    layer_idx: int,
) -> Callable:
    """
    Create a forward hook that offloads the layer after computation.
    """
    def hook(module, args, output):
        offload_manager.layer_forward_complete(layer_idx)
        return output

    return hook


def install_offload_hooks(
    pipe,
    offload_manager: LayerOffloadManager,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Install forward hooks for automatic offloading.

    This is less invasive than wrapping modules.

    Returns:
        List of hook handles (for removal if needed)
    """
    handles = []

    # Install hooks on double stream blocks
    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        pre_hook = create_offloaded_forward_hook(offload_manager, idx)
        post_hook = create_offloaded_forward_post_hook(offload_manager, idx)

        handles.append(block.register_forward_pre_hook(pre_hook))
        handles.append(block.register_forward_hook(post_hook))

    # Install hooks on single stream blocks
    num_double = len(pipe.transformer.transformer_blocks)
    for idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        layer_idx = num_double + idx
        pre_hook = create_offloaded_forward_hook(offload_manager, layer_idx)
        post_hook = create_offloaded_forward_post_hook(offload_manager, layer_idx)

        handles.append(block.register_forward_pre_hook(pre_hook))
        handles.append(block.register_forward_hook(post_hook))

    logger.info(f"Installed {len(handles)} offload hooks")

    return handles


@contextmanager
def offloaded_inference(
    pipe,
    config: Optional[OffloadConfig] = None,
):
    """
    Context manager for running inference with offloading.

    Usage:
        with offloaded_inference(pipe) as manager:
            output = pipe(prompt=..., ...)
            manager.print_statistics()
    """
    if config is None:
        config = OffloadConfig()

    # Setup offloading
    manager = setup_offloading_for_pipeline(pipe, config)

    # Install hooks
    handles = install_offload_hooks(pipe, manager)

    try:
        yield manager
    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()

        # Print statistics
        if config.verbose:
            manager.print_statistics()


# Convenience function
def enable_offloading(
    pipe,
    use_pinned_memory: bool = True,
    enable_prefetch: bool = True,
    num_layers_on_gpu: int = 6,
    max_memory_gb: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[LayerOffloadManager, List]:
    """
    Enable dynamic offloading for a pipeline.

    IMPORTANT: For best memory efficiency, call pre_encode_and_offload() BEFORE
    this function to pre-compute prompt embeddings and free text encoder memory.

    Args:
        pipe: HunyuanVideoPipeline
        use_pinned_memory: Use pinned CPU memory for faster transfers
        enable_prefetch: Enable async prefetching
        num_layers_on_gpu: Number of layers to keep on GPU (sliding window size)
                          Higher = faster but more VRAM
                          Recommended: 4-8 for 24GB, 10-15 for 40GB+
        max_memory_gb: If set, auto-tune num_layers_on_gpu to fit this budget
                       e.g., 20.0 for 24GB GPU with headroom
        verbose: Print debug information

    Returns:
        Tuple of (LayerOffloadManager, list of hook handles)

    Usage:
        # Step 1: Pre-encode prompt (text encoders temporarily on GPU)
        prompt_embeds = pre_encode_and_offload(pipe, "your prompt here")

        # Step 2: Enable offloading (text encoders now on CPU)
        manager, handles = enable_offloading(pipe, num_layers_on_gpu=8)

        # Step 3: Run pipeline with pre-computed embeddings
        output = pipe(
            prompt_embeds=prompt_embeds['prompt_embeds'],
            pooled_prompt_embeds=prompt_embeds['pooled_prompt_embeds'],
            prompt_attention_mask=prompt_embeds['prompt_attention_mask'],
            ...
        )
        manager.print_statistics()
    """
    config = OffloadConfig(
        use_pinned_memory=use_pinned_memory,
        enable_prefetch=enable_prefetch,
        num_layers_on_gpu=num_layers_on_gpu,
        max_memory_gb=max_memory_gb,
        verbose=verbose,
    )

    manager = setup_offloading_for_pipeline(pipe, config)
    handles = install_offload_hooks(pipe, manager)

    return manager, handles
