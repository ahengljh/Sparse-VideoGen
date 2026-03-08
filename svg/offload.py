"""
Dynamic Layer Offloading for Video Diffusion Models

Provides CPU-GPU memory management for running large video diffusion models
(like HunyuanVideo) on consumer GPUs with limited VRAM.

Key features:
1. Sliding window of N transformer layers on GPU at a time
2. Async prefetching via CUDA streams to hide transfer latency
3. Pinned CPU memory for faster transfers
"""

import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from .logger import logger
from .timer import time_logging_decorator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OffloadConfig:
    """Configuration for dynamic layer offloading."""

    compute_device: str = "cuda"
    offload_device: str = "cpu"

    use_pinned_memory: bool = True
    num_layers_on_gpu: int = 6
    enable_prefetch: bool = True
    prefetch_count: int = 2

    empty_cache_frequency: int = 10

    # Auto-tune budget
    max_memory_gb: Optional[float] = None
    auto_tune_layers_on_gpu: bool = False
    max_memory_fraction: float = 0.90
    activation_reserve_gb: float = 4.0
    cuda_overhead_gb: float = 0.5
    auto_tune_allow_increase: bool = False

    verbose: bool = False


# ---------------------------------------------------------------------------
# Layer Offload Manager
# ---------------------------------------------------------------------------

class LayerOffloadManager:
    """
    Manages dynamic CPU-GPU offloading for transformer layers using a
    sliding window with async prefetch.
    """

    def __init__(
        self,
        transformer: nn.Module,
        config: OffloadConfig,
        double_blocks_attr: str = "transformer_blocks",
        single_blocks_attr: str = "single_transformer_blocks",
    ):
        self.transformer = transformer
        self.config = config

        self.double_blocks = getattr(transformer, double_blocks_attr, [])
        self.single_blocks = getattr(transformer, single_blocks_attr, [])
        self.all_blocks = list(self.double_blocks) + list(self.single_blocks)
        self.num_layers = len(self.all_blocks)

        self._layer_memory_mb = self._estimate_layer_memory()

        # Single source of truth for which layers are on GPU
        self._layers_on_gpu: set = set()
        self._layer_pinned: Dict[int, bool] = {}

        # CUDA stream for async prefetch
        self._prefetch_stream: Optional[torch.cuda.Stream] = None

        # Prefetch tracking
        self._prefetch_in_progress: Dict[int, bool] = {}
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}

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

    # -- Helpers -------------------------------------------------------------

    def _estimate_layer_memory(self) -> float:
        """Estimate memory per layer in MB."""
        if not self.all_blocks:
            return 0.0
        layer = self.all_blocks[0]
        total = sum(p.numel() * p.element_size() for p in layer.parameters())
        total += sum(b.numel() * b.element_size() for b in layer.buffers() if b is not None)
        return total / (1024 * 1024)

    def _auto_tune_layers_on_gpu(self):
        """Auto-tune the number of layers to keep on GPU based on memory budget."""
        if not torch.cuda.is_available():
            return
        if self.config.max_memory_gb is None and not self.config.auto_tune_layers_on_gpu:
            return

        # Determine budget
        if self.config.max_memory_gb is not None:
            budget_gb = float(self.config.max_memory_gb)
        else:
            total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
            fraction = max(0.50, min(0.98, float(self.config.max_memory_fraction)))
            budget_gb = total_gb * fraction

        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            budget_gb = min(budget_gb, allocated_gb + free_bytes / (1024**3))
        except Exception:
            pass

        reserve_gb = float(self.config.activation_reserve_gb) + float(self.config.cuda_overhead_gb)
        available_gb = budget_gb - allocated_gb - reserve_gb

        if available_gb <= 0:
            self.config.num_layers_on_gpu = 1
            self.config.prefetch_count = 0
            logger.warning(
                f"Auto-tune: insufficient budget (budget={budget_gb:.1f}GB, "
                f"allocated={allocated_gb:.1f}GB, reserve={reserve_gb:.1f}GB). "
                f"Forcing num_layers_on_gpu=1."
            )
            return

        if self._layer_memory_mb > 0:
            max_layers = int((available_gb * 1024) / self._layer_memory_mb)
            suggested = max(1, min(max_layers, self.num_layers))

            if self.config.auto_tune_allow_increase:
                tuned = suggested
            else:
                tuned = min(self.config.num_layers_on_gpu, suggested) if self.config.num_layers_on_gpu > 0 else suggested

            self.config.num_layers_on_gpu = tuned
            if tuned <= 1:
                self.config.prefetch_count = 0
            else:
                self.config.prefetch_count = min(self.config.prefetch_count, tuned - 1)

            logger.info(
                f"Auto-tuned layers on GPU: {tuned} "
                f"(~{self._layer_memory_mb:.1f}MB/layer, budget={budget_gb:.1f}GB, "
                f"allocated={allocated_gb:.1f}GB, reserve={reserve_gb:.1f}GB)"
            )

    # -- Pinned memory -------------------------------------------------------

    def _pin_layer_params(self, layer_idx: int):
        """Convert CPU layer parameters to pinned memory for faster GPU transfers."""
        layer = self.all_blocks[layer_idx]
        for param in layer.parameters():
            if param.device.type == 'cpu' and not param.data.is_pinned():
                pinned = torch.empty_like(param.data, pin_memory=True)
                pinned.copy_(param.data)
                param.data = pinned
        self._layer_pinned[layer_idx] = True

    # -- Move layers ---------------------------------------------------------

    def _move_layer_to_cpu(self, layer_idx: int):
        """Move a layer to CPU, optionally with pinned memory."""
        layer = self.all_blocks[layer_idx]

        # Clear sparse weight caches (not captured by .to('cpu'))
        if getattr(layer, "_has_mlp_2of4_sparsity", False):
            try:
                from .sparsity import clear_module_sparse_cache
                clear_module_sparse_cache(layer)
            except Exception:
                pass

        if self.config.use_pinned_memory:
            for param in layer.parameters():
                if param.device.type != 'cpu':
                    cpu_data = param.data.cpu()
                    if not cpu_data.is_pinned():
                        pinned = torch.empty_like(cpu_data, pin_memory=True)
                        pinned.copy_(cpu_data)
                        cpu_data = pinned
                    param.data = cpu_data

            # Handle buffers via sub-module traversal for correctness with nested modules
            for module in layer.modules():
                for name, buf in list(module._buffers.items()):
                    if buf is not None and buf.device.type != 'cpu':
                        cpu_buf = buf.cpu()
                        if not cpu_buf.is_pinned():
                            pinned = torch.empty_like(cpu_buf, pin_memory=True)
                            pinned.copy_(cpu_buf)
                            cpu_buf = pinned
                        module._buffers[name] = cpu_buf

            self._layer_pinned[layer_idx] = True
        else:
            layer.to('cpu')
            self._layer_pinned[layer_idx] = False

        self._layers_on_gpu.discard(layer_idx)

    def _move_layer_to_gpu(self, layer_idx: int, non_blocking: bool = False):
        """Move a layer to GPU."""
        layer = self.all_blocks[layer_idx]
        layer.to(self.config.compute_device, non_blocking=non_blocking)

        if getattr(layer, "_has_mlp_2of4_sparsity", False):
            try:
                from .sparsity import prepare_module_for_sparse_inference
                prepare_module_for_sparse_inference(layer)
            except Exception:
                pass

        self._layers_on_gpu.add(layer_idx)
        self.stats['gpu_loads'] += 1

    def _offload_layer(self, layer_idx: int):
        """Offload a layer from GPU back to CPU."""
        if layer_idx not in self._layers_on_gpu:
            return
        self._move_layer_to_cpu(layer_idx)
        self.stats['gpu_offloads'] += 1

    # -- Prefetch ------------------------------------------------------------

    @time_logging_decorator("Level 4 - Layer prefetch")
    def _start_prefetch(self, layer_idx: int):
        """Start async prefetch of a layer."""
        if not self.config.enable_prefetch or layer_idx >= self.num_layers:
            return
        if layer_idx in self._layers_on_gpu:
            return
        if self._prefetch_in_progress.get(layer_idx, False):
            return

        self._prefetch_in_progress[layer_idx] = True
        event = torch.cuda.Event()
        self._prefetch_events[layer_idx] = event

        with torch.cuda.stream(self._prefetch_stream):
            self._move_layer_to_gpu(layer_idx, non_blocking=True)
            event.record()

    def _wait_for_prefetch(self, layer_idx: int):
        """Wait for prefetch of a layer to complete."""
        if layer_idx in self._prefetch_events:
            self._prefetch_events[layer_idx].synchronize()
            del self._prefetch_events[layer_idx]
            self._prefetch_in_progress[layer_idx] = False
            self.stats['prefetch_hits'] += 1
        else:
            self.stats['prefetch_misses'] += 1

    # -- Sliding window eviction ---------------------------------------------

    def _evict_outside_window(self, keep_layer_idx: int):
        """Evict layers outside the current sliding window.

        Window extends forward by prefetch_count to avoid evicting
        layers that were just prefetched.
        """
        window_start = max(0, keep_layer_idx - self.config.num_layers_on_gpu + 1)
        # Extend window forward to protect prefetched layers
        window_end = keep_layer_idx + self.config.prefetch_count

        to_evict = [idx for idx in self._layers_on_gpu
                     if idx < window_start or idx > window_end]

        for idx in to_evict:
            self._offload_layer(idx)
            self.stats['window_slides'] += 1

    # -- Public API ----------------------------------------------------------

    def prepare_for_inference(self):
        """Set up CUDA streams, pin memory, and track layer locations."""
        if self._initialized:
            return

        logger.info(f"Preparing {self.num_layers} layers for offloaded inference...")
        self._auto_tune_layers_on_gpu()

        if self.config.enable_prefetch:
            self._prefetch_stream = torch.cuda.Stream()

        # Check layer locations and optionally pin CPU parameters
        for idx, layer in enumerate(self.all_blocks):
            first_param = next(layer.parameters(), None)
            if first_param is not None:
                on_gpu = first_param.device.type == 'cuda'
                if on_gpu:
                    self._layers_on_gpu.add(idx)
                elif self.config.use_pinned_memory:
                    self._pin_layer_params(idx)

        torch.cuda.empty_cache()
        self._initialized = True

        num_on_cpu = self.num_layers - len(self._layers_on_gpu)
        logger.info(f"Offload manager initialized. {num_on_cpu}/{self.num_layers} layers on CPU.")

    @time_logging_decorator("Level 3 - Ensure layer on GPU")
    def ensure_layer_on_gpu(self, layer_idx: int):
        """Ensure a layer is on GPU (load if necessary), evict old layers, prefetch next."""
        if not self._initialized:
            self.prepare_for_inference()

        if layer_idx in self._layers_on_gpu:
            # Already on GPU; still wait for in-flight prefetch to finish
            if self._prefetch_in_progress.get(layer_idx, False):
                self._wait_for_prefetch(layer_idx)
            # Evict even on early return so wrap-around (layer 0 after layer N-1)
            # does not leave stale layers on GPU.
            self._evict_outside_window(layer_idx)
            # Still prefetch upcoming layers (prefetch hit path)
            for i in range(1, self.config.prefetch_count + 1):
                next_idx = layer_idx + i
                if next_idx < self.num_layers:
                    self._start_prefetch(next_idx)
            return

        if self._prefetch_in_progress.get(layer_idx, False):
            self._wait_for_prefetch(layer_idx)
        else:
            self._move_layer_to_gpu(layer_idx, non_blocking=False)
            self.stats['prefetch_misses'] += 1

        # Evict layers outside sliding window
        self._evict_outside_window(layer_idx)

        # Prefetch upcoming layers
        for i in range(1, self.config.prefetch_count + 1):
            next_idx = layer_idx + i
            if next_idx < self.num_layers:
                self._start_prefetch(next_idx)

    @time_logging_decorator("Level 3 - Layer forward complete")
    def layer_forward_complete(self, layer_idx: int):
        """Called after a layer's forward pass. Periodically clears CUDA cache."""
        if (layer_idx + 1) % self.config.empty_cache_frequency == 0:
            torch.cuda.empty_cache()
            self.stats['cache_clears'] += 1

    # -- Statistics ----------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        total_ops = self.stats['gpu_loads'] + self.stats['prefetch_hits']
        prefetch_ratio = self.stats['prefetch_hits'] / total_ops if total_ops > 0 else 0.0
        return {**self.stats, 'prefetch_hit_ratio': prefetch_ratio, 'num_layers': self.num_layers}

    def print_statistics(self):
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


# ---------------------------------------------------------------------------
# Text Encoder Pre-encoding
# ---------------------------------------------------------------------------

def pre_encode_and_offload(
    pipe,
    prompt: str,
    prompt_2: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Pre-encode prompt using text encoders (one at a time on GPU), then
    offload them to CPU and free memory.

    Must be called BEFORE enable_offloading().

    Returns dict with prompt_embeds, pooled_prompt_embeds, prompt_attention_mask.
    """
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE

    initial_gpu_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    logger.info(f"GPU memory before encoding: {initial_gpu_gb:.2f}GB")

    prompt_embeds = None
    prompt_attention_mask = None
    pooled_prompt_embeds = None

    # Step 1: Encode with LLaMA (text_encoder, ~16GB bf16)
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        logger.info("Step 1/2: Encoding with LLaMA text_encoder...")
        enc = pipe.text_encoder
        enc.to(device)
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask = pipe._get_llama_prompt_embeds(
                prompt,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                num_videos_per_prompt=1,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )
        prompt_embeds = prompt_embeds.cpu()
        prompt_attention_mask = prompt_attention_mask.cpu()
        enc.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("  LLaMA encoding done, freed from GPU.")

    # Step 2: Encode with CLIP (text_encoder_2, ~0.4GB)
    if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
        logger.info("Step 2/2: Encoding with CLIP text_encoder_2...")
        enc2 = pipe.text_encoder_2
        enc2.to(device)
        with torch.no_grad():
            pooled_prompt_embeds = pipe._get_clip_prompt_embeds(
                prompt_2 if prompt_2 is not None else prompt,
                num_videos_per_prompt=1,
                device=device,
                dtype=dtype,
            )
        pooled_prompt_embeds = pooled_prompt_embeds.cpu()
        enc2.to('cpu')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("  CLIP encoding done, freed from GPU.")

    # Free text encoders from pipeline to release all GPU references
    for name in ['text_encoder', 'text_encoder_2', 'text_encoder_3']:
        if hasattr(pipe, name) and getattr(pipe, name) is not None:
            setattr(pipe, name, None)

    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    return {
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'prompt_attention_mask': prompt_attention_mask,
    }


# ---------------------------------------------------------------------------
# Pipeline Setup
# ---------------------------------------------------------------------------

def setup_offloading_for_pipeline(
    pipe,
    config: Optional[OffloadConfig] = None,
) -> LayerOffloadManager:
    """
    Set up dynamic offloading for a HunyuanVideo pipeline.

    Keeps transformer embedders/norms + VAE on GPU, transformer blocks on CPU.
    Call pre_encode_and_offload() BEFORE this to pre-compute prompt embeddings.
    """
    if config is None:
        config = OffloadConfig()

    logger.info("Setting up dynamic layer offloading for HunyuanVideo...")
    transformer = pipe.transformer

    offload_manager = LayerOffloadManager(
        transformer, config,
        double_blocks_attr="transformer_blocks",
        single_blocks_attr="single_transformer_blocks",
    )

    # Move non-block components to GPU (embedders, norms, projections)
    if torch.cuda.is_available():
        logger.info(f"  GPU memory before setup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    gpu_components = ['time_text_embed', 'x_embedder', 'context_embedder', 'norm_out', 'proj_out']
    if hasattr(transformer, 'rope'):
        gpu_components.append('rope')

    for comp_name in gpu_components:
        comp = getattr(transformer, comp_name, None)
        if comp is not None:
            comp.to(config.compute_device)
            first_param = next(comp.parameters(), None)
            if first_param is not None:
                size_mb = sum(p.numel() * p.element_size() for p in comp.parameters()) / 1024**2
                logger.info(f"  {comp_name}: {size_mb:.1f}MB -> {first_param.device}")

    # Ensure blocks stay on CPU
    transformer.transformer_blocks.to('cpu')
    transformer.single_transformer_blocks.to('cpu')
    logger.info(f"  transformer_blocks: {len(transformer.transformer_blocks)} blocks -> CPU")
    logger.info(f"  single_transformer_blocks: {len(transformer.single_transformer_blocks)} blocks -> CPU")

    torch.cuda.empty_cache()

    # Summary
    gpu_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters() if p.device.type == 'cuda')
    cpu_bytes = sum(p.numel() * p.element_size() for p in transformer.parameters() if p.device.type != 'cuda')
    logger.info(f"Transformer total: {gpu_bytes/1024**3:.2f}GB on GPU, {cpu_bytes/1024**3:.2f}GB on CPU")

    # VAE on GPU for decoding (~300MB)
    pipe.vae.to(config.compute_device)
    logger.info("VAE kept on GPU")

    # Text encoders should already be None/CPU from pre_encode_and_offload
    for enc_name in ['text_encoder', 'text_encoder_2', 'text_encoder_3']:
        enc = getattr(pipe, enc_name, None)
        if enc is not None:
            enc.to('cpu')
    logger.info("Text encoders on CPU (use pre-computed embeddings)")

    offload_manager.prepare_for_inference()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after setup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Register pre-hook to ensure embedders stay on GPU during forward
    embedder_names = list(gpu_components)

    def ensure_embedders_on_gpu(module, args):
        for name in embedder_names:
            comp = getattr(module, name, None)
            if comp is not None:
                first_param = next(comp.parameters(), None)
                if first_param is not None and first_param.device.type != 'cuda':
                    comp.to('cuda')
        return args

    transformer.register_forward_pre_hook(ensure_embedders_on_gpu)
    logger.info("Registered embedder-ensure hook on transformer")

    return offload_manager


def install_offload_hooks(
    pipe,
    offload_manager: LayerOffloadManager,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Install forward pre/post hooks on transformer blocks for automatic offloading.

    When block-level caching is active, the transformer forward loop skips
    cached blocks entirely (they never call block.forward()), so hooks on
    cached blocks simply never fire — no special logic needed here.
    """
    handles = []

    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        handles.append(block.register_forward_pre_hook(
            lambda mod, args, _idx=idx: (offload_manager.ensure_layer_on_gpu(_idx), args)[1]
        ))
        handles.append(block.register_forward_hook(
            lambda mod, args, out, _idx=idx: (offload_manager.layer_forward_complete(_idx), out)[1]
        ))

    num_double = len(pipe.transformer.transformer_blocks)
    for idx, block in enumerate(pipe.transformer.single_transformer_blocks):
        layer_idx = num_double + idx
        handles.append(block.register_forward_pre_hook(
            lambda mod, args, _idx=layer_idx: (offload_manager.ensure_layer_on_gpu(_idx), args)[1]
        ))
        handles.append(block.register_forward_hook(
            lambda mod, args, out, _idx=layer_idx: (offload_manager.layer_forward_complete(_idx), out)[1]
        ))

    logger.info(f"Installed {len(handles)} offload hooks")
    return handles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enable_offloading(
    pipe,
    use_pinned_memory: bool = True,
    enable_prefetch: bool = True,
    num_layers_on_gpu: int = 6,
    max_memory_gb: Optional[float] = None,
    auto_tune_layers_on_gpu: bool = False,
    max_memory_fraction: float = 0.90,
    activation_reserve_gb: float = 4.0,
    cuda_overhead_gb: float = 0.5,
    auto_tune_allow_increase: bool = False,
    verbose: bool = False,
) -> Tuple[LayerOffloadManager, List]:
    """
    Enable dynamic offloading for a pipeline.

    Call pre_encode_and_offload() BEFORE this to pre-compute prompt embeddings.

    Returns:
        Tuple of (LayerOffloadManager, list of hook handles)
    """
    config = OffloadConfig(
        use_pinned_memory=use_pinned_memory,
        enable_prefetch=enable_prefetch,
        num_layers_on_gpu=num_layers_on_gpu,
        max_memory_gb=max_memory_gb,
        auto_tune_layers_on_gpu=auto_tune_layers_on_gpu,
        max_memory_fraction=max_memory_fraction,
        activation_reserve_gb=activation_reserve_gb,
        cuda_overhead_gb=cuda_overhead_gb,
        auto_tune_allow_increase=auto_tune_allow_increase,
        verbose=verbose,
    )

    manager = setup_offloading_for_pipeline(pipe, config)
    handles = install_offload_hooks(pipe, manager)
    return manager, handles
