"""Temporal-group readout for WAM-Adapter.

The readout reuses the multi-layer representations produced by a single
Video-DiT forward pass over the clean current frame and the noisy future
latent slots.  It organizes those representations into Current / Early-Future /
Late-Future groups, applies group-specific cross-layer fusion with streaming
accumulation, and feeds a lightweight mixed-context action decoder that
directly regresses the action chunk.

Design details kept from the production path:

* every action block/head learns an independent cross-layer mixture for each
  temporal group (current, early future, late future);
* fused keys are normalized and expressed once in the action-side
  physical-time 1-D RoPE basis, aligned with the action timeline;
* fusion combines backbone layers, never tokens: temporal and spatial token
  positions stay separate, and the learned layer coefficients are
  unconstrained linear mixtures without softmax or other normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import sinusoidal_embedding_1d


def _validate_heads(hidden_dim: int, num_heads: int, name: str) -> int:
    if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
        raise ValueError(
            f"{name}: hidden_dim must be positive and divisible by num_heads, "
            f"got hidden_dim={hidden_dim}, num_heads={num_heads}."
        )
    head_dim = hidden_dim // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"{name}: rotary head_dim must be even, got {head_dim}.")
    return head_dim


def apply_rotary_1d(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply 1-D RoPE to ``[B, H, T, D]`` without touching other modalities."""
    if x.ndim != 4:
        raise ValueError(f"RoPE input must be [B,H,T,D], got {tuple(x.shape)}.")
    if positions.ndim != 1 or positions.shape[0] != x.shape[2]:
        raise ValueError(
            f"RoPE positions must be [T={x.shape[2]}], got {tuple(positions.shape)}."
        )
    dim = int(x.shape[-1])
    if dim % 2 != 0:
        raise ValueError(f"RoPE feature dimension must be even, got {dim}.")
    inv_freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dim, 2, device=x.device, dtype=torch.float32)
        / float(dim)
    )
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos = angles.cos().to(dtype=x.dtype)[None, None]
    sin = angles.sin().to(dtype=x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def undo_rotary_3d(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Undo Wan's already-applied 3-D RoPE on flattened ``[B,S,H*D]`` keys."""
    if x.ndim != 3:
        raise ValueError(f"3-D RoPE keys must be [B,S,D], got {tuple(x.shape)}.")
    if x.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Key width {x.shape[-1]} must be divisible by num_heads={num_heads}."
        )
    batch, length, width = x.shape
    head_dim = width // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"3-D RoPE head dimension must be even, got {head_dim}.")
    headed = x.reshape(batch, length, num_heads, head_dim)
    complex_x = torch.view_as_complex(
        headed.to(torch.float64).reshape(batch, length, num_heads, head_dim // 2, 2)
    )
    complex_freqs = freqs.to(device=x.device)
    if complex_freqs.device.type == "npu":
        complex_freqs = complex_freqs.to(torch.complex64)
    canonical = torch.view_as_real(complex_x * complex_freqs.conj()).flatten(2)
    return canonical.to(dtype=x.dtype)


@dataclass
class FusedContext:
    key: torch.Tensor
    value: torch.Tensor


class PerBlockPostRopeFusion(nn.Module):
    """Stream post-3D-RoPE video representations into one fused cache per action block."""

    capture_basis = "post_3d_rope"
    requires_rope_freqs = True

    def __init__(
        self,
        video_hidden_dim: int,
        video_num_heads: int,
        action_hidden_dim: int,
        action_num_heads: int,
        num_action_blocks: int,
        selected_layers: Sequence[int],
    ):
        super().__init__()
        if not selected_layers or len(set(selected_layers)) != len(selected_layers):
            raise ValueError("`selected_layers` must be a non-empty list of unique indices.")
        if num_action_blocks <= 0:
            raise ValueError("`num_action_blocks` must be positive.")
        self.video_hidden_dim = int(video_hidden_dim)
        self.video_num_heads = int(video_num_heads)
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_num_heads = int(action_num_heads)
        self.num_action_blocks = int(num_action_blocks)
        self.head_dim = _validate_heads(
            action_hidden_dim, action_num_heads, self.__class__.__name__
        )
        self.selected_layers = tuple(int(i) for i in selected_layers)
        self.layer_to_offset = {
            layer: offset for offset, layer in enumerate(self.selected_layers)
        }

        # Channel projections are shared across blocks. Only the inexpensive
        # cross-layer coefficients are block-specific.
        self.key_projection = nn.Linear(video_hidden_dim, action_hidden_dim)
        self.value_projection = nn.Linear(video_hidden_dim, action_hidden_dim)
        self.key_norms = nn.ModuleList(
            [nn.LayerNorm(self.head_dim) for _ in range(self.num_action_blocks)]
        )
        self.layer_weights = nn.Parameter(
            torch.full(
                (self.num_action_blocks, action_num_heads, len(self.selected_layers)),
                1.0 / float(len(self.selected_layers)),
            )
        )
        self._key_sum: Optional[torch.Tensor] = None
        self._value_sum: Optional[torch.Tensor] = None
        self._seen_layers: set[int] = set()

    def begin_capture(self):
        self._key_sum = None
        self._value_sum = None
        self._seen_layers = set()

    def project_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.layer_to_offset:
            raise ValueError(f"Layer {layer_idx} is not selected by this cross-layer fusion.")
        if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
            raise ValueError(
                f"Fused tensors must share [B,S,D], got {tuple(key.shape)} and {tuple(value.shape)}."
            )
        if key.shape[-1] != self.video_hidden_dim:
            raise ValueError(
                f"Fusion video hidden dim must be {self.video_hidden_dim}, got {key.shape[-1]}."
            )
        if freqs is None:
            raise ValueError("Post-RoPE capture requires the exact Wan 3-D RoPE frequencies.")

        canonical_key = undo_rotary_3d(key, freqs, self.video_num_heads)
        batch, length, _ = canonical_key.shape
        projected_key = self.key_projection(canonical_key).view(
            batch, length, self.action_num_heads, self.head_dim
        ).transpose(1, 2)
        projected_value = self.value_projection(value).view(
            batch, length, self.action_num_heads, self.head_dim
        ).transpose(1, 2)
        coefficient = self.layer_weights[
            :, :, self.layer_to_offset[layer_idx]
        ].to(device=key.device, dtype=key.dtype)[:, None, :, None, None]
        return (
            projected_key.unsqueeze(0) * coefficient,
            projected_value.unsqueeze(0) * coefficient,
        )

    def accumulate_projected(
        self,
        layer_idx: int,
        weighted_key: torch.Tensor,
        weighted_value: torch.Tensor,
    ):
        expected_rank = 5
        if weighted_key.ndim != expected_rank or weighted_key.shape != weighted_value.shape:
            raise ValueError(
                "Per-block projected tensors must share [A,B,H,S,D], got "
                f"{tuple(weighted_key.shape)} and {tuple(weighted_value.shape)}."
            )
        if weighted_key.shape[0] != self.num_action_blocks:
            raise ValueError("Projected action-block axis does not match the fusion module.")
        if layer_idx in self._seen_layers:
            raise RuntimeError(f"Layer {layer_idx} was captured more than once.")
        self._key_sum = weighted_key if self._key_sum is None else self._key_sum + weighted_key
        self._value_sum = (
            weighted_value if self._value_sum is None else self._value_sum + weighted_value
        )
        self._seen_layers.add(int(layer_idx))

    def accumulate(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ):
        weighted_key, weighted_value = self.project_layer(
            layer_idx, key, value, freqs=freqs
        )
        self.accumulate_projected(layer_idx, weighted_key, weighted_value)

    def finish_capture(self) -> tuple[FusedContext, ...]:
        missing = set(self.selected_layers) - self._seen_layers
        if missing:
            self.begin_capture()
            raise RuntimeError(f"Missing post-3D-RoPE representations from Wan layers: {sorted(missing)}.")
        if self._key_sum is None or self._value_sum is None:
            raise RuntimeError("No representations were captured by the per-block fusion.")

        # All current-frame visual tokens live at action-time origin 0. Applying
        # 1-D RoPE is therefore an identity numerically, but makes the coordinate
        # contract explicit and keeps it paired with action positions 1..T.
        positions = torch.zeros(
            self._key_sum.shape[3], device=self._key_sum.device, dtype=torch.float32
        )
        outputs = tuple(
            FusedContext(
                key=apply_rotary_1d(self.key_norms[i](self._key_sum[i]), positions),
                value=self._value_sum[i],
            )
            for i in range(self.num_action_blocks)
        )
        self.begin_capture()
        return outputs

    def finish_capture_groups(
        self, group_lengths: Sequence[int], group_positions: Optional[Sequence[float]] = None
    ) -> tuple[tuple[FusedContext, ...], ...]:
        """Finish a capture while preserving contiguous current/future groups.

        ``forward_backbone`` captures the concatenated video sequence.  The
        temporal-group readout uses the same capture, but the action decoder
        needs to know which tokens came from the clean current frame and which
        came from future slots.  ``group_positions`` assigns one action-time
        1-D RoPE position per group; defaults to ``0, 1, ...``.
        """
        missing = set(self.selected_layers) - self._seen_layers
        if missing:
            self.begin_capture()
            raise RuntimeError(f"Missing post-3D-RoPE representations from Wan layers: {sorted(missing)}.")
        if self._key_sum is None or self._value_sum is None:
            raise RuntimeError("No representations were captured by the per-block fusion.")
        if not group_lengths or any(int(n) <= 0 for n in group_lengths):
            raise ValueError("group_lengths must contain positive token counts.")
        total = int(sum(int(n) for n in group_lengths))
        if total != int(self._key_sum.shape[3]):
            raise ValueError(
                f"Temporal groups cover {total} tokens, but capture has "
                f"{int(self._key_sum.shape[3])}."
            )
        num_groups = len(group_lengths)
        if group_positions is None:
            positions_seq = [float(i) for i in range(num_groups)]
        else:
            if len(group_positions) != num_groups:
                raise ValueError(
                    f"group_positions must have one entry per group "
                    f"({num_groups}), got {len(group_positions)}."
                )
            positions_seq = [float(v) for v in group_positions]
        boundaries = []
        start = 0
        for length in group_lengths:
            end = start + int(length)
            boundaries.append((start, end))
            start = end
        outputs = []
        for group_idx, (start, end) in enumerate(boundaries):
            positions = torch.full(
                (end - start,),
                positions_seq[group_idx],
                device=self._key_sum.device,
                dtype=torch.float32,
            )
            group_outputs = tuple(
                FusedContext(
                    key=apply_rotary_1d(
                        self.key_norms[i](self._key_sum[i, :, :, start:end]), positions
                    ),
                    value=self._value_sum[i, :, :, start:end],
                )
                for i in range(self.num_action_blocks)
            )
            outputs.append(group_outputs)
        self.begin_capture()
        return tuple(outputs)


class MixedContextAttention(nn.Module):
    """One softmax over action context and the block-specific fused context."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = _validate_heads(hidden_dim, num_heads, self.__class__.__name__)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = float(dropout)

    def forward(
        self, x: torch.Tensor, context: FusedContext, positions: torch.Tensor
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, action_key, action_value = qkv.unbind(dim=2)
        q = apply_rotary_1d(q.transpose(1, 2), positions)
        action_key = apply_rotary_1d(action_key.transpose(1, 2), positions)
        action_value = action_value.transpose(1, 2)
        mixed_key = torch.cat([action_key, context.key], dim=2)
        mixed_value = torch.cat([action_value, context.value], dim=2)
        output = F.scaled_dot_product_attention(
            q,
            mixed_key,
            mixed_value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.hidden_dim)
        return self.out(output)


class MixedContextActionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.mixed_attn = MixedContextAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, context: FusedContext, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.mixed_attn(self.norm_attn(x), context, positions)
        return x + self.ffn(self.norm_ffn(x))


class ActionQueryDecoder(nn.Module):
    """Deterministic action-query decoder with per-block cross-layer fusion."""

    def __init__(
        self,
        action_dim: int,
        video_hidden_dim: int,
        video_num_heads: int,
        context_hidden_dim: int,
        selected_layers: Sequence[int],
        hidden_dim: int = 1024,
        num_heads: int = 16,
        ffn_dim: int = 4096,
        num_layers: int = 2,
        max_action_horizon: int = 32,
        position_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"Action decoder must contain at least one block, got {num_layers}.")
        if position_dim <= 0 or position_dim % 2 != 0:
            raise ValueError(f"`position_dim` must be positive and even, got {position_dim}.")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.position_dim = int(position_dim)
        self.num_heads = int(num_heads)
        self.head_dim = _validate_heads(hidden_dim, num_heads, self.__class__.__name__)
        self.fusion = PerBlockPostRopeFusion(
            video_hidden_dim=video_hidden_dim,
            video_num_heads=video_num_heads,
            action_hidden_dim=hidden_dim,
            action_num_heads=num_heads,
            num_action_blocks=num_layers,
            selected_layers=selected_layers,
        )
        self.action_queries = nn.Parameter(
            torch.randn(max_action_horizon, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.position_projection = nn.Sequential(
            nn.LayerNorm(position_dim),
            nn.Linear(position_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(context_hidden_dim), nn.Linear(context_hidden_dim, hidden_dim)
        )
        # Proprio projection is shared across action blocks to avoid duplicating
        # a large state interface; each block still has independent visual fusion.
        self.state_key = nn.Linear(hidden_dim, hidden_dim)
        self.state_value = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [MixedContextActionBlock(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, action_dim),
        )

    def _append_state_token(
        self, contexts: tuple[FusedContext, ...], state_context: torch.Tensor
    ) -> tuple[FusedContext, ...]:
        if state_context.ndim == 3:
            if state_context.shape[1] != 1:
                raise ValueError("Rank-3 state context must contain exactly one token.")
            state_context = state_context[:, 0]
        if state_context.ndim != 2:
            raise ValueError(
                f"`state_context` must be [B,D] or [B,1,D], got {tuple(state_context.shape)}."
            )
        state = self.state_projection(state_context)
        batch = state.shape[0]
        key = self.state_key(state).view(batch, self.num_heads, 1, self.head_dim)
        value = self.state_value(state).view(batch, self.num_heads, 1, self.head_dim)
        return tuple(
            FusedContext(
                key=torch.cat([context.key, key], dim=2),
                value=torch.cat([context.value, value], dim=2),
            )
            for context in contexts
        )

    def forward(self, action_horizon: int, state_context: torch.Tensor) -> torch.Tensor:
        if action_horizon <= 0 or action_horizon > self.max_action_horizon:
            raise ValueError(
                f"action_horizon must be in [1,{self.max_action_horizon}], got {action_horizon}."
            )
        contexts = self._append_state_token(self.fusion.finish_capture(), state_context)
        positions = torch.arange(
            1, action_horizon + 1, device=contexts[0].key.device, dtype=torch.float32
        )
        positional = sinusoidal_embedding_1d(self.position_dim, positions).to(
            dtype=contexts[0].key.dtype
        )
        x = self.action_queries[:action_horizon].to(dtype=contexts[0].key.dtype).unsqueeze(0)
        x = x.expand(state_context.shape[0], -1, -1)
        x = x + self.position_projection(positional).unsqueeze(0)
        for block, context in zip(self.blocks, contexts):
            x = block(x, context, positions)
        return self.action_head(self.final_norm(x))


class GroupedContextActionDecoder(ActionQueryDecoder):
    """Action decoder that consumes separate Current / Early-Future / Late-Future groups."""

    token_pooling_type = "grouped_context_post_rope"

    def __init__(self, *args, num_temporal_groups: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if num_temporal_groups <= 0:
            raise ValueError("num_temporal_groups must be positive")
        self.num_temporal_groups = int(num_temporal_groups)

    def forward(
        self,
        action_horizon: int,
        state_context: torch.Tensor,
        grouped_contexts: Optional[tuple[tuple[FusedContext, ...], ...]] = None,
    ) -> torch.Tensor:
        if grouped_contexts is None:
            return super().forward(action_horizon=action_horizon, state_context=state_context)
        if len(grouped_contexts) == 0:
            raise ValueError("Temporal-group decoder requires at least one group.")
        if len(grouped_contexts) != self.num_temporal_groups:
            raise ValueError(
                f"Expected {self.num_temporal_groups} temporal groups, got {len(grouped_contexts)}."
            )
        if any(len(group) != len(self.blocks) for group in grouped_contexts):
            raise ValueError("Every temporal group must provide one context per action block.")
        merged = []
        for block_idx in range(len(self.blocks)):
            keys = [group[block_idx].key for group in grouped_contexts]
            values = [group[block_idx].value for group in grouped_contexts]
            merged.append(FusedContext(key=torch.cat(keys, dim=2), value=torch.cat(values, dim=2)))
        contexts = self._append_state_token(tuple(merged), state_context)
        positions = torch.arange(
            1, action_horizon + 1, device=contexts[0].key.device, dtype=torch.float32
        )
        positional = sinusoidal_embedding_1d(self.position_dim, positions).to(
            dtype=contexts[0].key.dtype
        )
        x = self.action_queries[:action_horizon].to(dtype=contexts[0].key.dtype).unsqueeze(0)
        x = x.expand(state_context.shape[0], -1, -1)
        x = x + self.position_projection(positional).unsqueeze(0)
        for block, context in zip(self.blocks, contexts):
            x = block(x, context, positions)
        return self.action_head(self.final_norm(x))


class TemporalGroupCrossLayerFusion(nn.Module):
    """Stream canonical Video-DiT representations with group-specific all-layer mixtures."""

    capture_basis = "canonical"
    requires_rope_freqs = False

    def __init__(
        self,
        video_hidden_dim: int,
        video_num_heads: int,
        action_hidden_dim: int,
        action_num_heads: int,
        num_action_blocks: int,
        num_temporal_groups: int,
        selected_layers: Sequence[int],
        share_temporal_group_layer_weights: bool = False,
    ):
        super().__init__()
        if not selected_layers or len(set(selected_layers)) != len(selected_layers):
            raise ValueError("`selected_layers` must be a non-empty list of unique indices.")
        if num_action_blocks <= 0:
            raise ValueError("`num_action_blocks` must be positive.")
        if num_temporal_groups <= 0:
            raise ValueError("`num_temporal_groups` must be positive.")

        self.video_hidden_dim = int(video_hidden_dim)
        self.video_num_heads = int(video_num_heads)
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_num_heads = int(action_num_heads)
        self.num_action_blocks = int(num_action_blocks)
        self.num_temporal_groups = int(num_temporal_groups)
        self.share_temporal_group_layer_weights = bool(
            share_temporal_group_layer_weights
        )
        self.head_dim = _validate_heads(
            action_hidden_dim, action_num_heads, self.__class__.__name__
        )
        self.selected_layers = tuple(int(i) for i in selected_layers)
        self.layer_to_offset = {
            layer: offset for offset, layer in enumerate(self.selected_layers)
        }

        # The expensive channel projections are shared.  Only the lightweight
        # layer coefficients vary across action block, head, and temporal group.
        self.key_projection = nn.Linear(video_hidden_dim, action_hidden_dim)
        self.value_projection = nn.Linear(video_hidden_dim, action_hidden_dim)
        self.key_norms = nn.ModuleList(
            [nn.LayerNorm(self.head_dim) for _ in range(self.num_action_blocks)]
        )
        parameter_groups = (
            1 if self.share_temporal_group_layer_weights else self.num_temporal_groups
        )
        self.layer_weights = nn.Parameter(
            torch.full(
                (
                    self.num_action_blocks,
                    self.action_num_heads,
                    parameter_groups,
                    len(self.selected_layers),
                ),
                1.0 / float(len(self.selected_layers)),
            )
        )

        self._group_lengths: Optional[tuple[int, ...]] = None
        self._key_sum: Optional[torch.Tensor] = None
        self._value_sum: Optional[torch.Tensor] = None
        self._seen_layers: set[int] = set()
        # When set, the fusion captures only the first ``current_only_tokens``
        # video tokens (the clean current frame) and never exposes any future
        # token representation to the action decoder.  The backbone still runs
        # the full joint current+future sequence for the video objective; only
        # the read-out is restricted to the current frame.  This is the
        # "current-frame only" ablation and is orthogonal to future slots in
        # the video sequence.
        self.current_only_tokens: Optional[int] = None

    def prepare_group_capture(self, group_lengths: Sequence[int]) -> None:
        lengths = tuple(int(v) for v in group_lengths)
        if len(lengths) != self.num_temporal_groups:
            raise ValueError(
                f"Expected {self.num_temporal_groups} temporal groups, got {len(lengths)}."
            )
        if any(v <= 0 for v in lengths):
            raise ValueError("Every temporal group must contain at least one token.")
        self._group_lengths = lengths

    def begin_capture(self) -> None:
        # Keep the group layout prepared by the model; clear only tensors from
        # the preceding forward.
        self._key_sum = None
        self._value_sum = None
        self._seen_layers = set()

    def _token_group_indices(self, length: int, device: torch.device) -> torch.Tensor:
        if self._group_lengths is None:
            raise RuntimeError("Call prepare_group_capture() before the Wan forward.")
        if sum(self._group_lengths) != length:
            raise ValueError(
                f"Prepared groups cover {sum(self._group_lengths)} tokens, "
                f"but this layer contains {length}."
            )
        return torch.repeat_interleave(
            torch.arange(self.num_temporal_groups, device=device),
            torch.tensor(self._group_lengths, device=device),
        )

    def project_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del freqs
        if layer_idx not in self.layer_to_offset:
            raise ValueError(f"Layer {layer_idx} is not selected by this cross-layer fusion.")
        if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
            raise ValueError(
                f"Fused tensors must share [B,S,D], got {tuple(key.shape)} and "
                f"{tuple(value.shape)}."
            )
        if key.shape[-1] != self.video_hidden_dim:
            raise ValueError(
                f"Fusion video hidden dim must be {self.video_hidden_dim}, "
                f"got {key.shape[-1]}."
            )

        batch, length, _ = key.shape
        token_groups = self._token_group_indices(length, key.device)
        projected_key = self.key_projection(key).view(
            batch, length, self.action_num_heads, self.head_dim
        ).transpose(1, 2)
        projected_value = self.value_projection(value).view(
            batch, length, self.action_num_heads, self.head_dim
        ).transpose(1, 2)

        # No softmax: coefficients are unconstrained learnable linear mixtures.
        # Gather the appropriate group coefficient for every visual token.
        layer_coefficients = self.layer_weights[
            :, :, :, self.layer_to_offset[layer_idx]
        ]
        if self.share_temporal_group_layer_weights:
            # Preserve the current/early/late token groups and physical RoPE
            # positions while broadcasting one cross-layer readout to all.
            coefficients = layer_coefficients.expand(
                -1, -1, self.num_temporal_groups
            ).index_select(2, token_groups)
        else:
            coefficients = layer_coefficients.index_select(2, token_groups)
        coefficients = coefficients.to(device=key.device, dtype=key.dtype)
        coefficients = coefficients[:, None, :, :, None]
        return (
            projected_key.unsqueeze(0) * coefficients,
            projected_value.unsqueeze(0) * coefficients,
        )

    def accumulate_projected(
        self,
        layer_idx: int,
        weighted_key: torch.Tensor,
        weighted_value: torch.Tensor,
    ) -> None:
        if weighted_key.ndim != 5 or weighted_key.shape != weighted_value.shape:
            raise ValueError(
                "Projected tensors must share [A,B,H,S,D], got "
                f"{tuple(weighted_key.shape)} and {tuple(weighted_value.shape)}."
            )
        if weighted_key.shape[0] != self.num_action_blocks:
            raise ValueError("Projected action-block axis does not match the fusion module.")
        if layer_idx in self._seen_layers:
            raise RuntimeError(f"Layer {layer_idx} was captured more than once.")
        self._key_sum = weighted_key if self._key_sum is None else self._key_sum + weighted_key
        self._value_sum = (
            weighted_value
            if self._value_sum is None
            else self._value_sum + weighted_value
        )
        self._seen_layers.add(int(layer_idx))

    def accumulate(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> None:
        if self.current_only_tokens is not None:
            # "Current-frame only" ablation: expose only the clean current
            # frame's tokens (the sequence prefix) to the action decoder.
            # Future tokens still flow through the backbone for the video
            # objective.
            key = key[:, : self.current_only_tokens]
            value = value[:, : self.current_only_tokens]
        weighted_key, weighted_value = self.project_layer(
            layer_idx, key, value, freqs=freqs
        )
        self.accumulate_projected(layer_idx, weighted_key, weighted_value)

    def _validate_complete_capture(self) -> None:
        missing = set(self.selected_layers) - self._seen_layers
        if missing:
            self.begin_capture()
            raise RuntimeError(f"Missing canonical representations from Wan layers: {sorted(missing)}.")
        if self._key_sum is None or self._value_sum is None:
            raise RuntimeError("No representations were captured by the temporal-group fusion.")

    def finish_capture(self) -> tuple[FusedContext, ...]:
        """Current-only fallback with every token placed at physical time zero."""
        self._validate_complete_capture()
        positions = torch.zeros(
            self._key_sum.shape[3], device=self._key_sum.device, dtype=torch.float32
        )
        outputs = tuple(
            FusedContext(
                key=apply_rotary_1d(self.key_norms[i](self._key_sum[i]), positions),
                value=self._value_sum[i],
            )
            for i in range(self.num_action_blocks)
        )
        self.begin_capture()
        return outputs

    def finish_capture_groups(
        self,
        group_lengths: Sequence[int],
        group_positions: Optional[Sequence[float] | torch.Tensor] = None,
    ) -> tuple[tuple[FusedContext, ...], ...]:
        """Return groups after applying a real position to every temporal slot."""
        self._validate_complete_capture()
        lengths = tuple(int(v) for v in group_lengths)
        if lengths != self._group_lengths:
            raise ValueError(
                f"Finish groups {lengths} differ from prepared groups "
                f"{self._group_lengths}."
            )
        total = sum(lengths)
        if total != int(self._key_sum.shape[3]):
            raise ValueError(
                f"Temporal groups cover {total} tokens, but capture has "
                f"{int(self._key_sum.shape[3])}."
            )
        if group_positions is None:
            raise ValueError("Per-token physical positions are required for this fusion.")
        positions = torch.as_tensor(
            group_positions, device=self._key_sum.device, dtype=torch.float32
        ).flatten()
        if positions.numel() != total:
            raise ValueError(
                f"Expected one physical position for each of {total} tokens, "
                f"got {positions.numel()}."
            )

        outputs = []
        start = 0
        for length in lengths:
            end = start + length
            group_outputs = tuple(
                FusedContext(
                    key=apply_rotary_1d(
                        self.key_norms[i](self._key_sum[i, :, :, start:end]),
                        positions[start:end],
                    ),
                    value=self._value_sum[i, :, :, start:end],
                )
                for i in range(self.num_action_blocks)
            )
            outputs.append(group_outputs)
            start = end
        self.begin_capture()
        return tuple(outputs)


class WAMAdapterReadout(GroupedContextActionDecoder):
    """Temporal-group readout with independent block/head/group cross-layer mixtures."""

    token_pooling_type = "temporal_group_readout"

    def __init__(
        self,
        *args,
        num_temporal_groups: int = 3,
        share_temporal_group_layer_weights: bool = False,
        **kwargs,
    ):
        super().__init__(*args, num_temporal_groups=num_temporal_groups, **kwargs)
        old_fusion = self.fusion
        self.fusion = TemporalGroupCrossLayerFusion(
            video_hidden_dim=old_fusion.video_hidden_dim,
            video_num_heads=old_fusion.video_num_heads,
            action_hidden_dim=old_fusion.action_hidden_dim,
            action_num_heads=old_fusion.action_num_heads,
            num_action_blocks=old_fusion.num_action_blocks,
            num_temporal_groups=num_temporal_groups,
            selected_layers=old_fusion.selected_layers,
            share_temporal_group_layer_weights=share_temporal_group_layer_weights,
        )
