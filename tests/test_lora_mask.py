from __future__ import annotations

import pytest
import torch
from torch import nn

from wam_adapter.models.wan22.wan_video_dit import LoRALinearAdapter
from wam_adapter.utils.lora_mask import (
    install_wan_lora_output_mask,
    parse_layer_selection,
    parse_module_selection,
)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_q = LoRALinearAdapter(4, 4, rank=2, alpha=2)
        self.lora_k = LoRALinearAdapter(4, 4, rank=2, alpha=2)
        self.lora_v = LoRALinearAdapter(4, 4, rank=2, alpha=2)
        self.lora_o = LoRALinearAdapter(4, 4, rank=2, alpha=2)
        for module in self.children():
            nn.init.ones_(module.down_proj.weight)
            nn.init.ones_(module.up_proj.weight)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.cross_attn = FakeAttention()
        self.ffn_lora_in = LoRALinearAdapter(4, 8, rank=2, alpha=2)
        self.ffn_lora_out = LoRALinearAdapter(8, 4, rank=2, alpha=2)
        for module in (self.ffn_lora_in, self.ffn_lora_out):
            nn.init.ones_(module.down_proj.weight)
            nn.init.ones_(module.up_proj.weight)


class FakeVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": FakeVideo()})


def test_parse_selections():
    assert parse_layer_selection("0-2,4", num_layers=5) == {0, 1, 2, 4}
    assert parse_module_selection("cross_attn.k,cross_attn.v,ffn.0") == {
        "cross_attn.k",
        "cross_attn.v",
        "ffn.0",
    }


def test_mask_zeros_only_selected_lora_outputs():
    model = FakeModel()
    x = torch.ones(1, 4)
    selected = model.mixtures["video"].blocks[0].cross_attn.lora_k
    untouched = model.mixtures["video"].blocks[1].cross_attn.lora_k
    assert torch.count_nonzero(selected(x)) > 0
    assert torch.count_nonzero(untouched(x)) > 0

    manifest = install_wan_lora_output_mask(
        model,
        layers="0",
        modules="cross_attn.k",
        num_layers=2,
    )
    assert len(manifest["matched"]) == 1
    assert torch.count_nonzero(selected(x)) == 0
    assert torch.count_nonzero(untouched(x)) > 0


def test_mask_rejects_partial_or_mismatched_selection():
    model = FakeModel()
    with pytest.raises(ValueError, match="requires both"):
        install_wan_lora_output_mask(model, layers="0", modules=None, num_layers=2)
    with pytest.raises(RuntimeError, match="expected"):
        install_wan_lora_output_mask(
            model,
            layers="0-2",
            modules="all",
            num_layers=3,
        )
