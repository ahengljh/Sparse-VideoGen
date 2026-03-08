import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformer3DModel,
    HunyuanVideoTransformerBlock,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from ...logger import logger
from ...timer import time_logging_decorator


class HunyuanVideoSingleTransformerBlock_Sparse(HunyuanVideoSingleTransformerBlock):
    """Single Stream Transformer Block"""

    @time_logging_decorator("Level 0 Hunyuan Single Block - forward")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        with time_logging_decorator("Level 1 Single - norm & modulate"):
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        with time_logging_decorator("Level 1 Single - proj_act_mlp"):
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        with time_logging_decorator("Level 1 Single - attn"):
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                timestep=timestep,
            )
            attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        with time_logging_decorator("Level 1 Single - Concat and Linear"):
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            hidden_states = self.proj_out(hidden_states)

        with time_logging_decorator("Level 1 Single - gate & add"):
            hidden_states = gate.unsqueeze(1) * hidden_states
            hidden_states = hidden_states + residual

            hidden_states, encoder_hidden_states = (
                hidden_states[:, :-text_seq_length, :],
                hidden_states[:, -text_seq_length:, :],
            )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock_Sparse(HunyuanVideoTransformerBlock):
    """Double Stream Transformer Block"""

    @time_logging_decorator("Level 0 Hunyuan Double Block - forward")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        with time_logging_decorator("Level 1 Double - norm & modulate 1"):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # 2. Joint attention
        with time_logging_decorator("Level 1 Double - attn"):
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=freqs_cis,
                timestep=timestep,
            )

        # 3. Modulation and residual connection
        with time_logging_decorator("Level 1 Double - gate & add 1"):
            hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
            encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        with time_logging_decorator("Level 1 Double - norm & modulate 2"):
            norm_hidden_states = self.norm2(hidden_states)
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        with time_logging_decorator("Level 1 Double - ffn"):
            ff_output = self.ff(norm_hidden_states)
            context_ff_output = self.ff_context(norm_encoder_hidden_states)

        with time_logging_decorator("Level 1 Double - gate & add 2"):
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformer3DModel_Sparse(HunyuanVideoTransformer3DModel):
    """Hunyuan Video Transformer 3D Model"""

    # Block-level output caching config (set externally via VideoKReuseConfig)
    _block_cache_cfg = None  # VideoKReuseConfig or None
    _block_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    _block_cache_step: int = -1
    _block_cache_last_timestep = None
    _block_cache_hits: int = 0
    _block_cache_misses: int = 0
    _block_cache_timing: List[Dict] = []

    def _block_cache_step_idx(self, timestep) -> int:
        """Track diffusion step from timestep changes."""
        if timestep is None:
            return self._block_cache_step
        if isinstance(timestep, torch.Tensor):
            t_val = int(timestep[0].item()) if timestep.numel() > 0 else int(timestep.item())
        else:
            t_val = int(timestep)
        if self._block_cache_last_timestep is None or t_val != self._block_cache_last_timestep:
            self._block_cache_step += 1
            self._block_cache_last_timestep = t_val
        return self._block_cache_step

    def _is_reuse_step(self, step_idx: int) -> bool:
        """Check if this step should reuse cached outputs (vs compute fresh)."""
        cfg = self._block_cache_cfg
        if cfg is None or not cfg.enabled:
            return False
        if step_idx < cfg.start_step or cfg.interval <= 1:
            return False
        return (step_idx % cfg.interval) != 0

    def _is_store_step(self, step_idx: int) -> bool:
        """Check if this step should store outputs to cache."""
        cfg = self._block_cache_cfg
        if cfg is None or not cfg.enabled:
            return False
        return step_idx >= cfg.warmup_steps

    def reset_block_cache(self):
        """Reset block cache state between inference runs."""
        self._block_cache = {}
        self._block_cache_step = -1
        self._block_cache_last_timestep = None
        self._block_cache_hits = 0
        self._block_cache_misses = 0
        self._block_cache_timing = []

    def get_block_cache_stats(self) -> dict:
        return {"hits": self._block_cache_hits, "misses": self._block_cache_misses}

    def get_block_cache_timing(self) -> list:
        return list(self._block_cache_timing)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        first_frame_num_tokens = 1 * post_patch_height * post_patch_width

        # Ensure all input tensors are on the correct compute device (CUDA)
        # The pipeline may pass CPU tensors, but we need everything on GPU for computation
        # Get target device from embedder weights (which should be on CUDA)
        if hasattr(self, "time_text_embed"):
            first_param = next(self.time_text_embed.parameters(), None)
            if first_param is not None:
                device = first_param.device
            else:
                device = hidden_states.device
        else:
            device = hidden_states.device

        # If device is still CPU, force CUDA (this shouldn't happen if setup is correct)
        if device.type == "cpu":
            device = torch.device("cuda:0")
            logger.warning("Forcing device to cuda:0 (embedders were on CPU)")

        # Move ALL inputs to the compute device (silent - happens every step)
        hidden_states = hidden_states.to(device)
        timestep = timestep.to(device)
        if pooled_projections is not None:
            pooled_projections = pooled_projections.to(device)
        if guidance is not None:
            guidance = guidance.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)
        encoder_attention_mask = encoder_attention_mask.to(device)

        # Ensure all embedder modules are on the compute device
        # (They should already be, but this is a safety check)
        for module_name in ["time_text_embed", "x_embedder", "context_embedder", "rope", "norm_out", "proj_out"]:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    first_param = next(module.parameters(), None)
                    if first_param is not None and first_param.device != device:
                        logger.warning(f"{module_name} on {first_param.device}, moving to {device}")
                        module.to(device)

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, N]
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
        mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

        else:
            all_blocks = list(self.transformer_blocks) + list(self.single_transformer_blocks)
            num_blocks = len(all_blocks)
            cfg = self._block_cache_cfg
            step_idx = self._block_cache_step_idx(timestep) if cfg is not None and cfg.enabled else -1
            reuse = self._is_reuse_step(step_idx)
            store = self._is_store_step(step_idx)

            # Guard: only reuse if cache is populated (prevents crash on
            # edge-case configs where first reuse step has no prior store).
            if reuse and self._block_cache:
                # Reuse step: return the last cached block's output directly.
                # Only the final block's output feeds into norm_out/proj_out,
                # so we just need that one transfer from CPU→GPU.
                _t0 = time.perf_counter()
                last_cached = max(self._block_cache.keys())
                cached_hs, cached_enc = self._block_cache[last_cached]
                hidden_states = cached_hs.to(hidden_states.device)
                encoder_hidden_states = cached_enc.to(hidden_states.device)
                self._block_cache_hits += num_blocks
                if cfg.metrics_enabled:
                    torch.cuda.synchronize()
                    self._block_cache_timing.append({
                        "step": step_idx, "layer": -1, "action": "reuse_all",
                        "time_ms": (time.perf_counter() - _t0) * 1000,
                    })
            else:
                # Compute step: run all blocks, cache the last block's output.
                for layer_idx, block in enumerate(all_blocks):
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        timestep,
                        token_replace_emb,
                        first_frame_num_tokens,
                    )
                    if store and layer_idx == num_blocks - 1:
                        # Only cache the last block — it's the only one used on reuse.
                        self._block_cache[layer_idx] = (
                            hidden_states.detach().cpu(),
                            encoder_hidden_states.detach().cpu(),
                        )
                self._block_cache_misses += num_blocks

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


def replace_sparse_forward():
    HunyuanVideoSingleTransformerBlock.forward = HunyuanVideoSingleTransformerBlock_Sparse.forward
    HunyuanVideoTransformerBlock.forward = HunyuanVideoTransformerBlock_Sparse.forward

    HunyuanVideoTransformer3DModel.forward = HunyuanVideoTransformer3DModel_Sparse.forward

    # Patch block-level caching methods and attributes onto the base class
    for attr in [
        '_block_cache_cfg', '_block_cache', '_block_cache_step',
        '_block_cache_last_timestep', '_block_cache_hits', '_block_cache_misses',
        '_block_cache_timing',
        '_block_cache_step_idx', '_is_reuse_step',
        '_is_store_step', 'reset_block_cache', 'get_block_cache_stats',
        'get_block_cache_timing',
    ]:
        setattr(HunyuanVideoTransformer3DModel, attr, getattr(HunyuanVideoTransformer3DModel_Sparse, attr))
