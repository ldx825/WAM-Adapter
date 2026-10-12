"""Inference-only masks for diagnosing Wan LoRA importance.

The mask is implemented with forward hooks on ``LoRALinearAdapter`` outputs.
It therefore removes only the selected low-rank residual updates and never
mutates either the checkpoint or the frozen base projections.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


CANONICAL_MODULES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)

MODULE_ALIASES = {
    "all": set(CANONICAL_MODULES),
    "self_attn": {name for name in CANONICAL_MODULES if name.startswith("self_attn.")},
    "cross_attn": {name for name in CANONICAL_MODULES if name.startswith("cross_attn.")},
    "ffn": {"ffn.0", "ffn.2"},
}

_LORA_NAME_PATTERN = re.compile(
    r"(?:^|\.)blocks\.(?P<layer>\d+)\."
    r"(?P<module>self_attn\.lora_[qkvo]|cross_attn\.lora_[qkvo]|"
    r"ffn_lora_in|ffn_lora_out)$"
)


def parse_layer_selection(spec: Any, *, num_layers: int) -> set[int]:
    """Parse ``all``, comma-separated ids, and inclusive ranges such as 0-9."""
    if spec is None or str(spec).strip().lower() in {"", "none", "null"}:
        return set()
    if isinstance(spec, Iterable) and not isinstance(spec, str):
        layers = {int(item) for item in spec}
    else:
        text = str(spec).strip().lower()
        if text == "all":
            return set(range(num_layers))
        layers: set[int] = set()
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError(f"Invalid descending LoRA layer range: {token}")
                layers.update(range(start, end + 1))
            else:
                layers.add(int(token))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"LoRA mask layers outside [0, {num_layers - 1}]: {invalid}")
    return layers


def parse_module_selection(spec: Any) -> set[str]:
    """Parse canonical module names and convenient diagnostic aliases."""
    if spec is None or str(spec).strip().lower() in {"", "none", "null"}:
        return set()
    tokens = spec if isinstance(spec, Iterable) and not isinstance(spec, str) else str(spec).split(",")
    modules: set[str] = set()
    for raw_token in tokens:
        token = str(raw_token).strip().lower()
        if not token:
            continue
        if token in MODULE_ALIASES:
            modules.update(MODULE_ALIASES[token])
        elif token in CANONICAL_MODULES:
            modules.add(token)
        else:
            raise ValueError(
                f"Unsupported LoRA mask module '{token}'. Expected aliases "
                f"{sorted(MODULE_ALIASES)} or modules {list(CANONICAL_MODULES)}."
            )
    return modules


def _canonical_module(raw_module: str) -> str:
    if raw_module.startswith("self_attn.lora_"):
        return "self_attn." + raw_module.rsplit("_", 1)[-1]
    if raw_module.startswith("cross_attn.lora_"):
        return "cross_attn." + raw_module.rsplit("_", 1)[-1]
    if raw_module == "ffn_lora_in":
        return "ffn.0"
    if raw_module == "ffn_lora_out":
        return "ffn.2"
    raise ValueError(f"Unsupported LoRA module name: {raw_module}")


def install_wan_lora_output_mask(
    model: nn.Module,
    *,
    layers: Any,
    modules: Any,
    num_layers: int = 30,
) -> dict[str, Any]:
    """Zero selected Wan LoRA residual outputs and return an audit manifest.

    Layer and module selections are intersected.  Both must be non-empty when
    a mask is requested, which avoids accidentally disabling every LoRA module.
    """
    selected_layers = parse_layer_selection(layers, num_layers=num_layers)
    selected_modules = parse_module_selection(modules)
    if not selected_layers and not selected_modules:
        return {"enabled": False, "layers": [], "modules": [], "matched": []}
    if not selected_layers or not selected_modules:
        raise ValueError("LoRA masking requires both non-empty layer and module selections.")

    handles = []
    matched = []

    def zero_output(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(output)

    for name, module in model.named_modules():
        match = _LORA_NAME_PATTERN.search(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        canonical = _canonical_module(match.group("module"))
        if layer not in selected_layers or canonical not in selected_modules:
            continue
        handles.append(module.register_forward_hook(zero_output))
        matched.append({"name": name, "layer": layer, "module": canonical})

    expected = len(selected_layers) * len(selected_modules)
    if len(matched) != expected:
        for handle in handles:
            handle.remove()
        raise RuntimeError(
            f"LoRA mask matched {len(matched)} modules, expected {expected}. "
            "This usually indicates a model/config mismatch."
        )

    # Keep hooks alive for the lifetime of the evaluated model.
    model._wan_lora_mask_handles = handles  # type: ignore[attr-defined]
    return {
        "enabled": True,
        "layers": sorted(selected_layers),
        "modules": sorted(selected_modules),
        "matched": matched,
    }
