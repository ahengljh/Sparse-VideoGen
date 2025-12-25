"""
Layer Offloading for Memory-Efficient Video Diffusion Inference

This module enables running large video diffusion models (e.g., HunyuanVideo ~13B params)
on consumer GPUs (24GB) by dynamically moving transformer layers between CPU and GPU.

Key Features:
1. Sequential layer offloading: Only keep 1-2 layers on GPU at a time
2. Asynchronous prefetching: Load next layer while current one executes
3. Integration with SADSA: Sparse attention further reduces memory
4. Pinned memory for fast CPU-GPU transfers

Memory Analysis for HunyuanVideo at 720p, 129 frames:
- Model parameters: ~13B params (~26GB in bf16)
- Per-layer params: ~300M params (~600MB in bf16)
- Attention KV cache: ~3-4GB at peak
- With offloading + SADSA: ~8-12GB GPU memory usage

Architecture:
- 20 double transformer blocks + 40 single transformer blocks
- Process sequentially with layer-wise CPU offloading
- Prefetch next layer using CUDA streams for latency hiding
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable
from collections import OrderedDict
from contextlib import contextmanager
import threading
import gc

import torch
import torch.nn as nn

from .logger import logger
from .timer import time_logging_decorator


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OffloadConfig:
    """Configuration for layer offloading."""

    # Number of layers to keep on GPU simultaneously (1=minimum memory, 2=better latency)
    layers_on_gpu: int = 1

    # Use pinned (page-locked) memory for faster CPU-GPU transfers
    use_pinned_memory: bool = True

    # Use asynchronous prefetching with CUDA streams
    async_prefetch: bool = True

    # Synchronize after each layer (for debugging)
    sync_each_layer: bool = False

    # Keep certain critical layers on GPU permanently (first/last layers)
    pin_first_n_layers: int = 0
    pin_last_n_layers: int = 0

    # Aggressive memory cleanup
    aggressive_cleanup: bool = True

    # Verbose logging
    verbose: bool = False


# =============================================================================
# Memory Utilities
# =============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    try:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1e9
    except:
        free_gb = 0

    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free_gb,
    }


def move_to_device(
    module: nn.Module,
    device: torch.device,
    non_blocking: bool = True,
    stream: Optional[torch.cuda.Stream] = None,
) -> nn.Module:
    """Move module to device, optionally using a specific CUDA stream."""
    if stream is not None:
        with torch.cuda.stream(stream):
            module.to(device, non_blocking=non_blocking)
    else:
        module.to(device, non_blocking=non_blocking)
    return module


def pin_module_memory(module: nn.Module) -> nn.Module:
    """Pin module's parameters and buffers to page-locked memory for faster transfers."""
    for param in module.parameters():
        if param.is_cuda:
            continue
        if not param.data.is_pinned():
            param.data = param.data.pin_memory()

    for buffer in module.buffers():
        if buffer.is_cuda:
            continue
        if not buffer.data.is_pinned():
            buffer.data = buffer.data.pin_memory()

    return module


def cleanup_gpu_memory():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Layer State Manager
# =============================================================================

class LayerStateManager:
    """
    Manages the state of transformer layers for offloading.

    Tracks which layers are on CPU vs GPU and handles movement.
    """

    def __init__(
        self,
        layers: List[nn.Module],
        config: OffloadConfig,
    ):
        self.layers = layers
        self.config = config
        self.num_layers = len(layers)

        # Track layer locations: 'cpu' or 'cuda'
        self.layer_device: List[str] = ['cpu'] * self.num_layers

        # Layers currently on GPU
        self.gpu_layer_indices: List[int] = []

        # Prefetch stream for async transfers
        self.prefetch_stream: Optional[torch.cuda.Stream] = None
        if config.async_prefetch and torch.cuda.is_available():
            self.prefetch_stream = torch.cuda.Stream()

        # Pin memory if configured
        if config.use_pinned_memory:
            logger.info("[LayerOffload] Pinning layer memory for fast transfers...")
            for layer in layers:
                if not any(p.is_cuda for p in layer.parameters()):
                    pin_module_memory(layer)

        # Determine which layers to keep pinned on GPU
        self.pinned_gpu_layers: set = set()
        for i in range(min(config.pin_first_n_layers, self.num_layers)):
            self.pinned_gpu_layers.add(i)
        for i in range(max(0, self.num_layers - config.pin_last_n_layers), self.num_layers):
            self.pinned_gpu_layers.add(i)

        if self.pinned_gpu_layers:
            logger.info(f"[LayerOffload] Pinning layers {sorted(self.pinned_gpu_layers)} on GPU")

    def is_on_gpu(self, layer_idx: int) -> bool:
        """Check if layer is currently on GPU."""
        return self.layer_device[layer_idx] == 'cuda'

    def move_to_gpu(self, layer_idx: int, blocking: bool = True):
        """Move layer to GPU."""
        if self.is_on_gpu(layer_idx):
            return

        layer = self.layers[layer_idx]

        if self.prefetch_stream and not blocking:
            move_to_device(layer, torch.device('cuda'), non_blocking=True, stream=self.prefetch_stream)
        else:
            move_to_device(layer, torch.device('cuda'), non_blocking=False)

        self.layer_device[layer_idx] = 'cuda'
        if layer_idx not in self.gpu_layer_indices:
            self.gpu_layer_indices.append(layer_idx)

        if self.config.verbose:
            logger.info(f"[LayerOffload] Moved layer {layer_idx} to GPU")

    def move_to_cpu(self, layer_idx: int):
        """Move layer to CPU."""
        if not self.is_on_gpu(layer_idx):
            return

        # Don't move pinned layers
        if layer_idx in self.pinned_gpu_layers:
            return

        layer = self.layers[layer_idx]
        move_to_device(layer, torch.device('cpu'), non_blocking=True)

        self.layer_device[layer_idx] = 'cpu'
        if layer_idx in self.gpu_layer_indices:
            self.gpu_layer_indices.remove(layer_idx)

        if self.config.verbose:
            logger.info(f"[LayerOffload] Moved layer {layer_idx} to CPU")

    def ensure_on_gpu(self, layer_idx: int):
        """Ensure layer is on GPU, evicting others if needed."""
        if self.is_on_gpu(layer_idx):
            return

        # Evict layers if we have too many on GPU
        while len(self.gpu_layer_indices) >= self.config.layers_on_gpu:
            # Find oldest layer that isn't pinned
            for old_idx in self.gpu_layer_indices:
                if old_idx != layer_idx and old_idx not in self.pinned_gpu_layers:
                    self.move_to_cpu(old_idx)
                    break
            else:
                break  # All layers are pinned or current layer

        self.move_to_gpu(layer_idx, blocking=True)

    def prefetch_next_layer(self, current_layer_idx: int):
        """Asynchronously prefetch the next layer."""
        if not self.prefetch_stream:
            return

        next_idx = current_layer_idx + 1
        if next_idx >= self.num_layers:
            return

        if self.is_on_gpu(next_idx):
            return

        # Async load to GPU using prefetch stream
        self.move_to_gpu(next_idx, blocking=False)

    def sync_prefetch(self):
        """Synchronize prefetch stream."""
        if self.prefetch_stream:
            self.prefetch_stream.synchronize()

    def initialize_pinned_layers(self):
        """Move pinned layers to GPU during initialization."""
        for idx in sorted(self.pinned_gpu_layers):
            self.move_to_gpu(idx, blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def cleanup(self):
        """Move all layers to CPU and clean up."""
        for idx in list(self.gpu_layer_indices):
            self.move_to_cpu(idx)
        if self.config.aggressive_cleanup:
            cleanup_gpu_memory()


# =============================================================================
# Offloaded Transformer Wrapper
# =============================================================================

class OffloadedTransformerBlocks(nn.Module):
    """
    Wrapper that handles layer offloading for transformer blocks.

    Replaces the standard sequential forward pass with one that
    dynamically loads/unloads layers.
    """

    def __init__(
        self,
        double_blocks: nn.ModuleList,
        single_blocks: nn.ModuleList,
        config: Optional[OffloadConfig] = None,
    ):
        super().__init__()

        self.double_blocks = double_blocks
        self.single_blocks = single_blocks
        self.config = config or OffloadConfig()

        # Combine all layers for unified management
        self.all_layers = list(double_blocks) + list(single_blocks)
        self.num_double = len(double_blocks)
        self.num_single = len(single_blocks)
        self.num_total = self.num_double + self.num_single

        # Initialize state manager
        self.state_manager = LayerStateManager(self.all_layers, self.config)

        # Statistics
        self._transfer_count = 0
        self._initialized = False

        logger.info(f"[LayerOffload] Initialized with {self.num_total} layers "
                   f"({self.num_double} double + {self.num_single} single)")
        logger.info(f"[LayerOffload] Config: layers_on_gpu={self.config.layers_on_gpu}, "
                   f"async_prefetch={self.config.async_prefetch}")

    def initialize(self):
        """Initialize offloading - move pinned layers to GPU."""
        if self._initialized:
            return

        # First, move all layers to CPU
        logger.info("[LayerOffload] Moving all layers to CPU first...")
        for layer in self.all_layers:
            layer.to('cpu')

        if self.config.aggressive_cleanup:
            cleanup_gpu_memory()

        # Pin memory for fast transfers
        if self.config.use_pinned_memory:
            logger.info("[LayerOffload] Setting up pinned memory...")
            for layer in self.all_layers:
                pin_module_memory(layer)

        # Move pinned layers to GPU
        self.state_manager.initialize_pinned_layers()

        self._initialized = True

        mem = get_gpu_memory_info()
        logger.info(f"[LayerOffload] Initialization complete. "
                   f"GPU memory: {mem['allocated']:.2f}GB allocated, {mem['free']:.2f}GB free")

    def forward_double_block(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single double block with offloading."""
        global_idx = layer_idx  # Double blocks are first

        # Ensure layer is on GPU
        self.state_manager.ensure_on_gpu(global_idx)

        # Prefetch next layer
        self.state_manager.prefetch_next_layer(global_idx)

        # Forward pass
        block = self.double_blocks[layer_idx]
        hidden_states, encoder_hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

        # Sync if configured
        if self.config.sync_each_layer:
            self.state_manager.sync_prefetch()
            torch.cuda.synchronize()

        return hidden_states, encoder_hidden_states

    def forward_single_block(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single (single-stream) block with offloading."""
        global_idx = self.num_double + layer_idx  # Single blocks come after double

        # Ensure layer is on GPU
        self.state_manager.ensure_on_gpu(global_idx)

        # Prefetch next layer
        self.state_manager.prefetch_next_layer(global_idx)

        # Forward pass
        block = self.single_blocks[layer_idx]
        hidden_states, encoder_hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

        # Sync if configured
        if self.config.sync_each_layer:
            self.state_manager.sync_prefetch()
            torch.cuda.synchronize()

        return hidden_states, encoder_hidden_states

    def forward_all_layers(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Any,
        timestep: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all transformer blocks with offloading.
        """
        if not self._initialized:
            self.initialize()

        # Double transformer blocks
        for i in range(self.num_double):
            hidden_states, encoder_hidden_states = self.forward_double_block(
                i,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                timestep,
                *args,
                **kwargs,
            )

        # Final sync before switching to single blocks
        self.state_manager.sync_prefetch()

        # Single transformer blocks
        for i in range(self.num_single):
            hidden_states, encoder_hidden_states = self.forward_single_block(
                i,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                timestep,
                *args,
                **kwargs,
            )

        # Final sync
        self.state_manager.sync_prefetch()

        # Cleanup after each diffusion step if aggressive
        if self.config.aggressive_cleanup:
            # Keep first layer on GPU for next step
            for idx in list(self.state_manager.gpu_layer_indices):
                if idx != 0 and idx not in self.state_manager.pinned_gpu_layers:
                    self.state_manager.move_to_cpu(idx)

        return hidden_states, encoder_hidden_states

    def cleanup(self):
        """Clean up all GPU memory."""
        self.state_manager.cleanup()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage."""
        return get_gpu_memory_info()


# =============================================================================
# Integration with HunyuanVideo Pipeline
# =============================================================================

def setup_layer_offloading(
    pipe,
    config: Optional[OffloadConfig] = None,
) -> OffloadedTransformerBlocks:
    """
    Set up layer offloading for a HunyuanVideo pipeline.

    Args:
        pipe: HunyuanVideoPipeline instance
        config: Offloading configuration

    Returns:
        OffloadedTransformerBlocks wrapper
    """
    config = config or OffloadConfig()

    transformer = pipe.transformer

    # Create offloaded wrapper
    offloaded = OffloadedTransformerBlocks(
        double_blocks=transformer.transformer_blocks,
        single_blocks=transformer.single_transformer_blocks,
        config=config,
    )

    # Store reference on transformer
    transformer._offloaded_blocks = offloaded

    return offloaded


def replace_transformer_forward_with_offload(pipe, offloaded: OffloadedTransformerBlocks):
    """
    Replace the transformer forward to use offloaded blocks.

    This modifies the transformer's forward method to use our offloading wrapper.
    """
    transformer = pipe.transformer
    original_forward = transformer.forward

    def offloaded_forward(
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs = None,
        return_dict: bool = True,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(transformer, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = transformer.config.patch_size, transformer.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = transformer.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = transformer.time_text_embed(timestep, pooled_projections, guidance)

        hidden_states = transformer.x_embedder(hidden_states)
        encoder_hidden_states = transformer.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        mask_indices = indices >= effective_sequence_length.unsqueeze(1)
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # Extract timestep value for passing to attention
        timestep_val = timestep[0].item() if timestep.dim() > 0 else timestep.item()

        # 4. Transformer blocks with offloading
        hidden_states, encoder_hidden_states = offloaded.forward_all_layers(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask,
            image_rotary_emb,
            timestep_val,
        )

        # 5. Output projection
        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(transformer, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    transformer.forward = offloaded_forward
    logger.info("[LayerOffload] Replaced transformer forward with offloaded version")

    return offloaded_forward


# =============================================================================
# High-Level Setup Function
# =============================================================================

def enable_layer_offloading(
    pipe,
    layers_on_gpu: int = 1,
    use_pinned_memory: bool = True,
    async_prefetch: bool = True,
    pin_first_n_layers: int = 0,
    pin_last_n_layers: int = 0,
    verbose: bool = False,
) -> OffloadedTransformerBlocks:
    """
    Enable layer offloading for a HunyuanVideo pipeline.

    This is the main entry point for enabling memory-efficient inference.

    Args:
        pipe: HunyuanVideoPipeline instance
        layers_on_gpu: Number of layers to keep on GPU (1=minimum memory, 2=better latency)
        use_pinned_memory: Use pinned memory for fast CPU-GPU transfers
        async_prefetch: Asynchronously prefetch next layer
        pin_first_n_layers: Keep first N layers on GPU permanently
        pin_last_n_layers: Keep last N layers on GPU permanently
        verbose: Enable verbose logging

    Returns:
        OffloadedTransformerBlocks instance

    Example:
        pipe = HunyuanVideoPipeline.from_pretrained(...)
        offloaded = enable_layer_offloading(pipe, layers_on_gpu=1)

        # Now run inference as normal
        output = pipe(prompt="...", ...)

        # Cleanup when done
        offloaded.cleanup()
    """
    config = OffloadConfig(
        layers_on_gpu=layers_on_gpu,
        use_pinned_memory=use_pinned_memory,
        async_prefetch=async_prefetch,
        pin_first_n_layers=pin_first_n_layers,
        pin_last_n_layers=pin_last_n_layers,
        verbose=verbose,
    )

    logger.info("=" * 60)
    logger.info("[LayerOffload] Setting up layer offloading for 24GB GPU inference")
    logger.info("=" * 60)

    # Get initial memory
    mem_before = get_gpu_memory_info()
    logger.info(f"[LayerOffload] Initial GPU memory: {mem_before['allocated']:.2f}GB allocated")

    # Setup offloading
    offloaded = setup_layer_offloading(pipe, config)

    # Replace forward with offloaded version
    replace_transformer_forward_with_offload(pipe, offloaded)

    # Initialize (moves layers to CPU, sets up pinned memory)
    offloaded.initialize()

    mem_after = get_gpu_memory_info()
    logger.info(f"[LayerOffload] After setup: {mem_after['allocated']:.2f}GB allocated, "
               f"saved {mem_before['allocated'] - mem_after['allocated']:.2f}GB")

    return offloaded
