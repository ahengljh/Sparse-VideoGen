"""
Training-free 2:4 MLP sparsity + outlier split.

This module is designed to be compatible with the project's dynamic offloading:
- We replace selected nn.Linear modules with a wrapper that keeps an outlier
  subset of output channels dense while enforcing 2:4 structured sparsity on
  the remaining rows.
- When semi-structured sparse support is available (PyTorch + CUDA), we can
  cache a semi-structured weight representation on GPU for faster matmul.
- The cached sparse weights are NOT registered as parameters/buffers to avoid
  device-transfer issues; callers (e.g., offload manager) must clear caches
  before moving modules back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import logger


def _semi_structured_supported() -> bool:
    """
    Best-effort check for PyTorch semi-structured sparsity support.

    On supported builds, torch.sparse.to_sparse_semi_structured exists and
    accepts CUDA fp16/bf16 weights with 2:4 sparsity pattern.
    """
    return bool(getattr(torch, "sparse", None) is not None and hasattr(torch.sparse, "to_sparse_semi_structured"))


@dataclass
class MLP2of4SparsityConfig:
    enabled: bool = True
    outlier_ratio: float = 0.02  # Keep this fraction of output channels dense
    outlier_min_channels: int = 0
    outlier_metric: str = "l2"  # {"l2", "maxabs"}
    use_semi_structured: bool = True  # Use torch.sparse.to_sparse_semi_structured if available
    # If semi-structured weights are prepared, keep the dense pruned weight on CPU to avoid
    # doubling GPU weight memory (important for 24GB GPUs).
    keep_dense_pruned_weight_on_gpu: bool = False
    verbose: bool = False


def _select_outlier_rows(weight: torch.Tensor, *, ratio: float, min_channels: int, metric: str) -> torch.Tensor:
    """
    Select outlier output channels (rows) to keep dense.

    Returns a 1D LongTensor of row indices.
    """
    out_features = weight.shape[0]
    if ratio <= 0 and min_channels <= 0:
        return torch.empty((0,), dtype=torch.long)

    k = int(round(out_features * ratio))
    k = max(k, int(min_channels))
    k = min(k, out_features)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long)

    w = weight.detach()
    if metric == "maxabs":
        scores = w.abs().float().amax(dim=1)
    else:
        # Default: L2 norm
        scores = w.float().pow(2).sum(dim=1)

    _, idx = torch.topk(scores, k=k, largest=True, sorted=False)
    return torch.sort(idx).values


def _prune_2of4_rows(weight: torch.Tensor) -> torch.Tensor:
    """
    Enforce 2:4 structured sparsity along the input-feature dimension.

    For every consecutive group of 4 weights in each output row, keep the 2
    largest-magnitude entries and zero the rest.
    """
    out_features, in_features = weight.shape
    if in_features % 4 != 0:
        raise ValueError(f"2:4 requires in_features multiple of 4, got {in_features}")

    w = weight.detach().contiguous()
    w_view = w.reshape(out_features, in_features // 4, 4)
    abs_view = w_view.abs().float()
    top2 = torch.topk(abs_view, k=2, dim=-1, largest=True, sorted=False).indices

    mask = torch.zeros_like(w_view, dtype=torch.bool)
    mask.scatter_(-1, top2, True)
    pruned = torch.where(mask, w_view, torch.zeros_like(w_view))
    return pruned.reshape(out_features, in_features).to(weight.dtype)


class OutlierSplit2of4Linear(nn.Module):
    """
    Linear layer with training-free outlier split + 2:4 sparsity.

    Output channels are partitioned into:
    - Dense "outlier" rows (kept fully dense)
    - Sparse rows (2:4 pruned). Optionally cached as semi-structured sparse
      weights on GPU when supported.
    """

    def __init__(self, base: nn.Linear, config: MLP2of4SparsityConfig):
        super().__init__()
        self.config = config
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.has_bias = base.bias is not None

        # Keep indices on CPU to avoid unnecessary GPU memory usage.
        outlier_idx = _select_outlier_rows(
            base.weight.detach(),
            ratio=config.outlier_ratio,
            min_channels=config.outlier_min_channels,
            metric=config.outlier_metric,
        ).cpu()
        self._outlier_idx_cpu = outlier_idx
        all_idx = torch.arange(self.out_features, dtype=torch.long)
        outlier_mask = torch.zeros(self.out_features, dtype=torch.bool)
        outlier_mask[outlier_idx] = True
        self._sparse_idx_cpu = all_idx[~outlier_mask]

        # Materialize weights as frozen Parameters so offloading/pinning logic
        # (which operates on parameters) continues to work.
        device = base.weight.device
        dtype = base.weight.dtype

        # Dense outliers
        outlier_weight = base.weight.detach()[self._outlier_idx_cpu.to(device)]
        self.outlier_weight = nn.Parameter(outlier_weight.to(dtype), requires_grad=False)
        if self.has_bias:
            outlier_bias = base.bias.detach()[self._outlier_idx_cpu.to(device)]
            self.outlier_bias = nn.Parameter(outlier_bias.to(dtype), requires_grad=False)
        else:
            self.outlier_bias = None

        # Sparse remainder (2:4 pruned)
        sparse_weight = base.weight.detach()[self._sparse_idx_cpu.to(device)]
        sparse_weight = _prune_2of4_rows(sparse_weight)
        self.sparse_weight = nn.Parameter(sparse_weight.to(dtype), requires_grad=False)
        if self.has_bias:
            sparse_bias = base.bias.detach()[self._sparse_idx_cpu.to(device)]
            self.sparse_bias = nn.Parameter(sparse_bias.to(dtype), requires_grad=False)
        else:
            self.sparse_bias = None

        # Cache for semi-structured sparse weights (GPU only).
        self._sparse_weight_ss = None
        self._sparse_weight_ss_device = None

        # Cache for GPU indices (lazily populated on first forward on GPU)
        self._sparse_idx_gpu = None
        self._outlier_idx_gpu = None
        self._indices_device = None

        if config.verbose:
            logger.info(
                f"[2:4] Wrapped Linear({self.in_features}->{self.out_features}) "
                f"outliers={len(self._outlier_idx_cpu)} sparse_rows={len(self._sparse_idx_cpu)}"
            )

    @property
    def outlier_count(self) -> int:
        return int(self._outlier_idx_cpu.numel())

    @property
    def sparse_count(self) -> int:
        return int(self._sparse_idx_cpu.numel())

    def clear_sparse_cache(self):
        """Free cached GPU semi-structured weights and indices (must be called before offloading to CPU)."""
        self._sparse_weight_ss = None
        self._sparse_weight_ss_device = None
        self._sparse_idx_gpu = None
        self._outlier_idx_gpu = None
        self._indices_device = None

    def prepare_sparse_cache(self):
        """
        Prepare semi-structured sparse weights and GPU indices.

        Safe to call repeatedly; will re-use cache when possible.
        """
        dev = self.sparse_weight.device if self.sparse_weight.device.type == "cuda" else None

        # Prepare GPU indices if on CUDA
        if dev is not None:
            self._ensure_gpu_indices(dev)

        # Skip semi-structured prep if not supported or not on CUDA
        if not self.config.use_semi_structured or not _semi_structured_supported():
            return
        if dev is None:
            return
        if self.sparse_weight.dtype not in (torch.float16, torch.bfloat16):
            return

        if self._sparse_weight_ss is not None and self._sparse_weight_ss_device == dev:
            return

        try:
            self._sparse_weight_ss = torch.sparse.to_sparse_semi_structured(self.sparse_weight.contiguous())
            self._sparse_weight_ss_device = dev
            if not self.config.keep_dense_pruned_weight_on_gpu:
                # Move dense pruned weight back to CPU to avoid doubling GPU weight memory.
                # Keep it pinned when possible for faster future transfers.
                cpu_tensor = self.sparse_weight.detach().to("cpu")
                if not cpu_tensor.is_pinned():
                    try:
                        pinned = torch.empty_like(cpu_tensor, pin_memory=True)
                        pinned.copy_(cpu_tensor)
                        cpu_tensor = pinned
                    except RuntimeError:
                        # Pinned memory allocation can fail on some systems
                        pass
                self.sparse_weight = nn.Parameter(cpu_tensor, requires_grad=False)
        except Exception as e:
            # Fall back to dense pruned weights.
            if self.config.verbose:
                logger.warning(f"[2:4] Semi-structured prepare failed; using dense pruned weight. Error: {e}")
            self._sparse_weight_ss = None
            self._sparse_weight_ss_device = None

    def _get_sparse_weight(self) -> torch.Tensor:
        if self._sparse_weight_ss is not None:
            return self._sparse_weight_ss
        return self.sparse_weight

    def _ensure_gpu_indices(self, device: torch.device) -> None:
        """Ensure indices are cached on the correct GPU device."""
        if self._indices_device == device:
            return
        # Cache indices on GPU for faster scatter operations
        self._sparse_idx_gpu = self._sparse_idx_cpu.to(device)
        self._outlier_idx_gpu = self._outlier_idx_cpu.to(device) if self.outlier_count > 0 else None
        self._indices_device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare cache lazily if possible.
        if self._sparse_weight_ss is None:
            self.prepare_sparse_cache()

        # Ensure GPU indices are ready on the correct device
        if x.device.type == "cuda":
            self._ensure_gpu_indices(x.device)

        y_sparse = F.linear(x, self._get_sparse_weight(), self.sparse_bias)
        if self.outlier_count > 0:
            y_out = F.linear(x, self.outlier_weight, self.outlier_bias)
        else:
            y_out = None

        # Reassemble in original output channel order.
        out_shape = (*y_sparse.shape[:-1], self.out_features)
        # Use zeros instead of empty to avoid uninitialized memory issues
        y = x.new_zeros(out_shape)

        # Use GPU indices for faster scatter if available
        sparse_idx = self._sparse_idx_gpu if self._sparse_idx_gpu is not None else self._sparse_idx_cpu
        outlier_idx = self._outlier_idx_gpu if self._outlier_idx_gpu is not None else self._outlier_idx_cpu

        if self.sparse_count > 0:
            y[..., sparse_idx] = y_sparse
        if y_out is not None:
            y[..., outlier_idx] = y_out
        return y


def apply_mlp_2of4_sparsity(
    module: nn.Module,
    config: MLP2of4SparsityConfig,
    *,
    include_name_tokens: Tuple[str, ...] = ("ff", "ff_context", "proj_mlp"),
    exclude_name_tokens: Tuple[str, ...] = ("attn", "to_q", "to_k", "to_v", "to_out", "add_q", "add_k", "add_v"),
) -> int:
    """
    Replace selected nn.Linear modules under `module` with OutlierSplit2of4Linear.

    Returns number of Linear modules replaced.
    """
    if not config.enabled:
        return 0

    replaced = 0

    def should_replace(full_name: str) -> bool:
        if not any(tok in full_name for tok in include_name_tokens):
            return False
        if any(tok in full_name for tok in exclude_name_tokens):
            return False
        return True

    def recurse(parent: nn.Module, prefix: str):
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and should_replace(full):
                try:
                    wrapped = OutlierSplit2of4Linear(child, config)
                except Exception as e:
                    if config.verbose:
                        logger.warning(f"[2:4] Skip {full}: {e}")
                    continue
                setattr(parent, child_name, wrapped)
                replaced += 1
            else:
                recurse(child, full)

    recurse(module, "")

    if replaced > 0:
        # Hint to the offload manager to prepare/clear caches on moves.
        setattr(module, "_has_mlp_2of4_sparsity", True)

    return replaced


def prepare_module_for_sparse_inference(module: nn.Module):
    """Prepare (cache) semi-structured weights for any OutlierSplit2of4Linear under module."""
    for m in module.modules():
        if isinstance(m, OutlierSplit2of4Linear):
            m.prepare_sparse_cache()


def clear_module_sparse_cache(module: nn.Module):
    """Clear cached semi-structured weights for any OutlierSplit2of4Linear under module."""
    for m in module.modules():
        if isinstance(m, OutlierSplit2of4Linear):
            m.clear_sparse_cache()
