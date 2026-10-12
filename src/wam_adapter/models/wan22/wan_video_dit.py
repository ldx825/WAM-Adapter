import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from einops import rearrange
from .helpers.gradient import gradient_checkpoint_forward

from wam_adapter.utils.logging_config import get_logger

logger = get_logger(__name__)

LORA_TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "self_attn.q": ("self_attn.q",),
    "self_attn.k": ("self_attn.k",),
    "self_attn.v": ("self_attn.v",),
    "self_attn.o": ("self_attn.o",),
    "cross_attn.q": ("cross_attn.q",),
    "cross_attn.k": ("cross_attn.k",),
    "cross_attn.v": ("cross_attn.v",),
    "cross_attn.o": ("cross_attn.o",),
    "ffn": ("ffn.0", "ffn.2"),
    "mlp": ("ffn.0", "ffn.2"),
    "ffn.0": ("ffn.0",),
    "ffn_in": ("ffn.0",),
    "ffn.2": ("ffn.2",),
    "ffn_out": ("ffn.2",),
}


def _normalize_lora_target_modules(
    target_modules: Optional[list[str] | tuple[str, ...] | str],
) -> tuple[str, ...]:
    if target_modules is None:
        raw_targets: list[str] = ["ffn.0", "ffn.2"]
    elif isinstance(target_modules, str):
        raw_targets = [item.strip() for item in target_modules.split(",")]
    else:
        raw_targets = [str(item).strip() for item in target_modules]

    normalized: list[str] = []
    seen = set()
    for item in raw_targets:
        if not item:
            continue
        key = item.strip().lower()
        if key not in LORA_TARGET_ALIASES:
            raise ValueError(
                f"Unsupported LoRA target module `{item}`. "
                f"Expected one of: {sorted(LORA_TARGET_ALIASES.keys())}"
            )
        for target_name in LORA_TARGET_ALIASES[key]:
            if target_name in seen:
                continue
            normalized.append(target_name)
            seen.add(target_name)
    if not normalized:
        raise ValueError("LoRA target modules must include at least one supported module.")
    return tuple(normalized)


class LoRALinearAdapter(nn.Module):
    """Low-rank residual adapter for a single linear projection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        if alpha <= 0.0:
            raise ValueError(f"`alpha` must be positive, got {alpha}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.down_proj = nn.Linear(self.in_dim, self.rank, bias=False)
        self.up_proj = nn.Linear(self.rank, self.out_dim, bias=False)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.down_proj(self.dropout(x))) * self.scale

    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.lora_q: Optional[LoRALinearAdapter] = None
        self.lora_k: Optional[LoRALinearAdapter] = None
        self.lora_v: Optional[LoRALinearAdapter] = None
        self.lora_o: Optional[LoRALinearAdapter] = None
        
        # self.attn = AttentionModule(self.num_heads)

    def configure_lora(
        self,
        target_modules: set[str],
        rank: int,
        alpha: float,
        dropout: float,
    ):
        if "q" in target_modules and self.lora_q is None:
            self.lora_q = LoRALinearAdapter(self.hidden_dim, self.attn_hidden_dim, rank, alpha, dropout)
        if "k" in target_modules and self.lora_k is None:
            self.lora_k = LoRALinearAdapter(self.hidden_dim, self.attn_hidden_dim, rank, alpha, dropout)
        if "v" in target_modules and self.lora_v is None:
            self.lora_v = LoRALinearAdapter(self.hidden_dim, self.attn_hidden_dim, rank, alpha, dropout)
        if "o" in target_modules and self.lora_o is None:
            self.lora_o = LoRALinearAdapter(self.attn_hidden_dim, self.hidden_dim, rank, alpha, dropout)

    def get_lora_modules(self) -> list[nn.Module]:
        return [module for module in (self.lora_q, self.lora_k, self.lora_v, self.lora_o) if module is not None]

    def forward(
        self,
        x,
        freqs,
        self_attn_mask: Optional[torch.Tensor] = None,
        return_canonical_repr: bool = False,
        return_post_rope_repr: bool = False,
    ):
        if return_canonical_repr and return_post_rope_repr:
            raise ValueError("Only one self-attention representation basis can be returned at a time.")
        q = self.q(x)
        if self.lora_q is not None:
            q = q + self.lora_q(x)
        q = self.norm_q(q)
        k = self.k(x)
        if self.lora_k is not None:
            k = k + self.lora_k(x)
        k = self.norm_k(k)
        v = self.v(x)
        if self.lora_v is not None:
            v = v + self.lora_v(x)
        q = rope_apply(q, freqs, self.num_heads)
        canonical_k = k
        k = rope_apply(canonical_k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        out = self.o(x)
        if self.lora_o is not None:
            out = out + self.lora_o(x)
        if return_canonical_repr:
            # `canonical_k` is exactly the normalized key immediately before
            # Wan's 3-D RoPE.  Returning the already-computed tensors avoids a
            # second projection in downstream readout interfaces.
            return out, canonical_k, v
        if return_post_rope_repr:
            # Reuse the exact key already rotated for Wan self-attention.  The
            # readout explicitly undoes this 3-D basis before action-side use.
            return out, k, v
        return out


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.lora_q: Optional[LoRALinearAdapter] = None
        self.lora_k: Optional[LoRALinearAdapter] = None
        self.lora_v: Optional[LoRALinearAdapter] = None
        self.lora_o: Optional[LoRALinearAdapter] = None
            
        # self.attn = AttentionModule(self.num_heads)

    def configure_lora(
        self,
        target_modules: set[str],
        rank: int,
        alpha: float,
        dropout: float,
    ):
        if "q" in target_modules and self.lora_q is None:
            self.lora_q = LoRALinearAdapter(self.hidden_dim, self.attn_hidden_dim, rank, alpha, dropout)
        if "k" in target_modules and self.lora_k is None:
            self.lora_k = LoRALinearAdapter(self.hidden_dim, self.attn_hidden_dim, rank, alpha, dropout)
        if "v" in target_modules and self.lora_v is None:
            self.lora_v = LoRALinearAdapter(self.hidden_dim, self.attn_hidden_dim, rank, alpha, dropout)
        if "o" in target_modules and self.lora_o is None:
            self.lora_o = LoRALinearAdapter(self.attn_hidden_dim, self.hidden_dim, rank, alpha, dropout)

    def get_lora_modules(self) -> list[nn.Module]:
        return [module for module in (self.lora_q, self.lora_k, self.lora_v, self.lora_o) if module is not None]

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.q(x)
        if self.lora_q is not None:
            q = q + self.lora_q(x)
        q = self.norm_q(q)
        k = self.k(ctx)
        if self.lora_k is not None:
            k = k + self.lora_k(ctx)
        k = self.norm_k(k)
        v = self.v(ctx)
        if self.lora_v is not None:
            v = v + self.lora_v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        out = self.o(x)
        if self.lora_o is not None:
            out = out + self.lora_o(x)
        return out


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class ResidualAdapter(nn.Module):
    """Lightweight bottleneck adapter applied as a residual correction."""

    def __init__(self, hidden_dim: int, adapter_dim: int, eps: float, scale: float = 1.0):
        super().__init__()
        if adapter_dim <= 0:
            raise ValueError(f"`adapter_dim` must be positive, got {adapter_dim}")
        self.scale = float(scale)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU(approximate="tanh")
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        # Zero-init keeps adapter-enabled checkpoints close to the frozen backbone at start.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.up_proj(self.act(self.down_proj(self.norm(x))))
        if self.scale != 1.0:
            delta = delta * self.scale
        return x + delta, delta

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()
        self.ffn_lora_in: Optional[LoRALinearAdapter] = None
        self.ffn_lora_out: Optional[LoRALinearAdapter] = None

    def configure_lora(
        self,
        target_modules: set[str],
        rank: int,
        alpha: float,
        dropout: float,
    ):
        self_attn_targets = {
            target_name.split(".", 1)[1]
            for target_name in target_modules
            if target_name.startswith("self_attn.")
        }
        if self_attn_targets:
            self.self_attn.configure_lora(self_attn_targets, rank, alpha, dropout)

        cross_attn_targets = {
            target_name.split(".", 1)[1]
            for target_name in target_modules
            if target_name.startswith("cross_attn.")
        }
        if cross_attn_targets:
            self.cross_attn.configure_lora(cross_attn_targets, rank, alpha, dropout)

        if "ffn.0" in target_modules and self.ffn_lora_in is None:
            self.ffn_lora_in = LoRALinearAdapter(self.hidden_dim, self.ffn_dim, rank, alpha, dropout)
        if "ffn.2" in target_modules and self.ffn_lora_out is None:
            self.ffn_lora_out = LoRALinearAdapter(self.ffn_dim, self.hidden_dim, rank, alpha, dropout)

    def get_lora_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        modules.extend(self.self_attn.get_lora_modules())
        modules.extend(self.cross_attn.get_lora_modules())
        if self.ffn_lora_in is not None:
            modules.append(self.ffn_lora_in)
        if self.ffn_lora_out is not None:
            modules.append(self.ffn_lora_out)
        return modules

    def has_lora(self) -> bool:
        return len(self.get_lora_modules()) > 0

    def enable_lora_training(self):
        for module in self.get_lora_modules():
            module.train()
            module.requires_grad_(True)

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        context_mask=None,
        self_attn_mask: Optional[torch.Tensor] = None,
        capture_self_attn_repr: bool = False,
        capture_self_attn_basis: str = "canonical",
    ):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if capture_self_attn_basis not in {"canonical", "post_3d_rope"}:
            raise ValueError(
                "`capture_self_attn_basis` must be canonical or post_3d_rope, "
                f"got {capture_self_attn_basis}."
            )
        self_attn_output = self.self_attn(
            input_x,
            freqs,
            self_attn_mask=self_attn_mask,
            return_canonical_repr=(
                capture_self_attn_repr and capture_self_attn_basis == "canonical"
            ),
            return_post_rope_repr=(
                capture_self_attn_repr and capture_self_attn_basis == "post_3d_rope"
            ),
        )
        if capture_self_attn_repr:
            self_attn_output, captured_k, captured_v = self_attn_output
        x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        ffn_hidden = self.ffn[0](input_x)
        if self.ffn_lora_in is not None:
            ffn_hidden = ffn_hidden + self.ffn_lora_in(input_x)
        ffn_hidden = self.ffn[1](ffn_hidden)
        ffn_out = self.ffn[2](ffn_hidden)
        if self.ffn_lora_out is not None:
            ffn_out = ffn_out + self.ffn_lora_out(ffn_hidden)
        x = self.gate(x, gate_mlp, ffn_out)
        if capture_self_attn_repr:
            return x, captured_k, captured_v
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_wam_adapter: bool = False,
        adapter_layer_indices: Optional[list[int]] = None,
        adapter_dim: int = 128,
        adapter_scale: float = 1.0,
        use_backbone_lora: bool = False,
        lora_layer_indices: Optional[list[int]] = None,
        lora_target_modules: Optional[list[str] | tuple[str, ...] | str] = None,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.use_wam_adapter = bool(use_wam_adapter)
        self.adapter_dim = int(adapter_dim)
        self.adapter_scale = float(adapter_scale)
        self.adapter_layer_indices = self._resolve_adapter_layer_indices(
            num_layers=num_layers,
            adapter_layer_indices=adapter_layer_indices,
        ) if self.use_wam_adapter else tuple()
        self.use_backbone_lora = bool(use_backbone_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = _normalize_lora_target_modules(lora_target_modules) if self.use_backbone_lora else tuple()
        if self.use_backbone_lora and self.lora_rank <= 0:
            raise ValueError(f"`lora_rank` must be positive when `use_backbone_lora=true`, got {self.lora_rank}")
        lora_default_layers = (
            list(self.adapter_layer_indices)
            if len(self.adapter_layer_indices) > 0
            else [8, 12, 16, 20, 24]
        )
        self.lora_layer_indices = self._resolve_adapter_layer_indices(
            num_layers=num_layers,
            adapter_layer_indices=lora_default_layers if lora_layer_indices is None else lora_layer_indices,
        ) if self.use_backbone_lora else tuple()

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.wam_adapters = nn.ModuleDict({
            str(layer_idx): ResidualAdapter(
                hidden_dim=hidden_dim,
                adapter_dim=self.adapter_dim,
                eps=eps,
                scale=self.adapter_scale,
            )
            for layer_idx in self.adapter_layer_indices
        })
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")
        self.reset_wam_adapter_cache()
        if self.use_backbone_lora:
            self.configure_backbone_lora()

    def configure_backbone_lora(self):
        if not self.use_backbone_lora:
            return
        target_modules = set(self.lora_target_modules)
        for layer_idx in self.lora_layer_indices:
            self.blocks[layer_idx].configure_lora(
                target_modules=target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
        logger.info(
            "Enabled backbone LoRA on layers=%s targets=%s rank=%d alpha=%.3f dropout=%.3f",
            list(self.lora_layer_indices),
            list(self.lora_target_modules),
            self.lora_rank,
            self.lora_alpha,
            self.lora_dropout,
        )

    def has_backbone_lora(self) -> bool:
        return self.use_backbone_lora and len(self.lora_layer_indices) > 0

    def get_backbone_lora_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        if not self.has_backbone_lora():
            return modules
        for layer_idx in self.lora_layer_indices:
            modules.extend(self.blocks[layer_idx].get_lora_modules())
        return modules

    def enable_backbone_lora_training(self):
        if not self.has_backbone_lora():
            return
        for layer_idx in self.lora_layer_indices:
            self.blocks[layer_idx].enable_lora_training()

    @staticmethod
    def _resolve_adapter_layer_indices(
        num_layers: int,
        adapter_layer_indices: Optional[list[int]],
    ) -> tuple[int, ...]:
        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}")
        if adapter_layer_indices is None or len(adapter_layer_indices) == 0:
            positions = (0.55, 0.70, 0.85, 1.0)
            adapter_layer_indices = [
                int(round((num_layers - 1) * position))
                for position in positions
            ]
        normalized = sorted({int(layer_idx) for layer_idx in adapter_layer_indices})
        if normalized[0] < 0 or normalized[-1] >= num_layers:
            raise ValueError(
                f"`adapter_layer_indices` must be within [0, {num_layers - 1}], got {normalized}"
            )
        return tuple(normalized)

    def reset_wam_adapter_cache(self):
        # Cache every selected adapter layer; the legacy single-layer API still reads the topmost one.
        self._wam_adapter_cache: dict[str, Any] = {
            "layers": {},
            "layer_idx": None,
            "backbone_tokens": None,
            "adapted_tokens": None,
        }

    def get_wam_action_fusion_states(
        self,
        fallback_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_tokens = self._wam_adapter_cache.get("backbone_tokens")
        adapted_tokens = self._wam_adapter_cache.get("adapted_tokens")
        if backbone_tokens is None or adapted_tokens is None:
            if fallback_tokens is None:
                raise ValueError(
                    "Adapter fusion states are unavailable; pass `fallback_tokens` when no adapter layer ran."
                )
            return fallback_tokens, fallback_tokens
        return backbone_tokens, adapted_tokens

    def get_wam_action_fusion_layer_states(
        self,
        selected_layers: Optional[list[int]] = None,
        fallback_tokens: Optional[torch.Tensor] = None,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        layer_cache = self._wam_adapter_cache.get("layers", {})
        if selected_layers is None:
            selected_layers = list(self.adapter_layer_indices)
        selected_layers = [int(layer_idx) for layer_idx in selected_layers]

        if not selected_layers:
            if fallback_tokens is None:
                raise ValueError("No adapter fusion layers are configured.")
            return [(0, fallback_tokens, fallback_tokens)]

        states: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for layer_idx in selected_layers:
            if layer_idx not in layer_cache:
                if fallback_tokens is None:
                    raise ValueError(
                        f"Adapter fusion state for layer {layer_idx} is unavailable in the current cache."
                    )
                states.append((layer_idx, fallback_tokens, fallback_tokens))
                continue
            layer_state = layer_cache[layer_idx]
            states.append(
                (
                    layer_idx,
                    layer_state["backbone_tokens"],
                    layer_state["adapted_tokens"],
                )
            )
        return states

    def apply_wam_adapter(
        self,
        layer_idx: int,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_wam_adapter:
            return tokens
        layer_key = str(int(layer_idx))
        if layer_key not in self.wam_adapters:
            return tokens
        backbone_tokens = tokens
        adapted_tokens, _ = self.wam_adapters[layer_key](backbone_tokens)
        layer_state = {
            "backbone_tokens": backbone_tokens,
            "adapted_tokens": adapted_tokens,
        }
        self._wam_adapter_cache["layers"][int(layer_idx)] = layer_state
        self._wam_adapter_cache = {
            **self._wam_adapter_cache,
            "layer_idx": int(layer_idx),
            "backbone_tokens": backbone_tokens,
            "adapted_tokens": adapted_tokens,
        }
        return adapted_tokens


    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")
            
            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim, 
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward_backbone(
        self,
        pre_state: Dict[str, Any],
        readout: Optional[Any] = None,
        gradient_checkpointing_override: Optional[bool] = None,
    ) -> torch.Tensor:
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]

        self.reset_wam_adapter_cache()
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None

        if readout is not None:
            readout.begin_capture()
            selected_readout_layers = set(int(i) for i in readout.selected_layers)
            capture_basis = str(getattr(readout, "capture_basis", "canonical"))
        else:
            selected_readout_layers = set()
            capture_basis = "canonical"
        use_gradient_checkpointing = (
            self.use_gradient_checkpointing
            if gradient_checkpointing_override is None
            else bool(gradient_checkpointing_override)
        )
        # Checkpointing has no benefit under no-grad inference and some torch
        # versions warn when it is invoked without a differentiable graph.
        use_gradient_checkpointing = bool(
            use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        )

        try:
            for layer_idx, block in enumerate(self.blocks):
                capture_layer = layer_idx in selected_readout_layers
                if use_gradient_checkpointing:
                    if capture_layer:
                        # Keep projection and layer weighting inside the same
                        # recomputed region.  Only the compact weighted
                        # contribution crosses the checkpoint boundary, rather
                        # than retaining every full-width canonical tensor.
                        def checkpointed_block_with_readout(
                            *args,
                            _block=block,
                            _layer_idx=layer_idx,
                            **kwargs,
                        ):
                            block_x, captured_k, captured_v = _block(
                                *args,
                                capture_self_attn_repr=True,
                                capture_self_attn_basis=capture_basis,
                                **kwargs,
                            )
                            weighted_k, weighted_v = readout.project_layer(
                                _layer_idx, captured_k, captured_v, freqs=freqs
                            )
                            return block_x, weighted_k, weighted_v

                        block_output = gradient_checkpoint_forward(
                            checkpointed_block_with_readout,
                            True,
                            x_tokens,
                            context_emb,
                            t_mod,
                            freqs,
                            context_mask=context_attn_mask,
                            self_attn_mask=self_attn_mask,
                        )
                    else:
                        block_output = gradient_checkpoint_forward(
                            block,
                            True,
                            x_tokens,
                            context_emb,
                            t_mod,
                            freqs,
                            context_mask=context_attn_mask,
                            self_attn_mask=self_attn_mask,
                            capture_self_attn_repr=False,
                            capture_self_attn_basis=capture_basis,
                        )
                else:
                    block_output = block(
                        x_tokens,
                        context_emb,
                        t_mod,
                        freqs,
                        context_mask=context_attn_mask,
                        self_attn_mask=self_attn_mask,
                        capture_self_attn_repr=capture_layer,
                        capture_self_attn_basis=capture_basis,
                    )
                if capture_layer:
                    if use_gradient_checkpointing:
                        x_tokens, weighted_k, weighted_v = block_output
                        # Accumulation remains outside recomputation and is the
                        # only stateful operation in the capture path.
                        readout.accumulate_projected(layer_idx, weighted_k, weighted_v)
                    else:
                        x_tokens, captured_k, captured_v = block_output
                        readout.accumulate(
                            layer_idx, captured_k, captured_v, freqs=freqs
                        )
                else:
                    x_tokens = block_output
                # When enabled, adapters only perturb selected mid/high layers and keep the default path unchanged.
                x_tokens = self.apply_wam_adapter(layer_idx=layer_idx, tokens=x_tokens)
        except Exception:
            if readout is not None:
                readout.begin_capture()
            raise
        return x_tokens

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = self.forward_backbone(pre_state)
        return self.post_dit(x_tokens, pre_state)
