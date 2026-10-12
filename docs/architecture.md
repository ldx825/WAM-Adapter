# Architecture and integration notes

Everything below was verified against a trained WAM-Adapter checkpoint and its
dumped config.

---

## 1. Where the model lives

| Piece | File |
|---|---|
| Model class | `src/wam_adapter/models/wan22/wam_adapter.py` (standalone `nn.Module`) |
| Temporal-group readout | `src/wam_adapter/models/wan22/wam_adapter_readout.py` |
| Hydra factory | `src/wam_adapter/runtime.py` → `create_wam_adapter` |
| Model config | `configs/model/wam_adapter.yaml` |
| Shared loader helpers | `src/wam_adapter/models/wan22/helpers/loader.py` |

The class owns the complete train/infer path so it can be stepped through in a
debugger:

```text
build_inputs → pre_dit → forward_backbone(readout=…) → finish_capture_groups → action decoder
```

The factory resolves configs and calls `WAMAdapter.from_wan22_pretrained(...)`:

```python
def create_wam_adapter(action_decoder_config=None, **kwargs):
    """Hydra entry point for WAM-Adapter."""
    from .models.wan22.wam_adapter import WAMAdapter

    return _create_wam_adapter_model(
        WAMAdapter,
        action_decoder_config=action_decoder_config,
        **kwargs,
    )
```

Canonical instantiation (what the 50-task run used):

```yaml
model:
  _target_: wam_adapter.runtime.create_wam_adapter
  action_decoder_config:
    num_future_slots: 2
    action_decoder_layers: 2
    readout_gradient_checkpointing: false
    group_positions: [0, 16, 32]
    temporal_position_stride: 16
    share_temporal_group_layer_weights: false
    cached_latent_spatial_downsample_factor: 2
    cached_latent_spatial_shape: [24, 20]
    readout_layer_indices: [0, 1, 2, …, 29]     # all 30 backbone blocks
```

## 2. Backbone shape (the config numbers are not the final ones)

`configs/model/*.yaml` still carries a `video_dit_config` block with `hidden_dim: 3072`,
`ffn_dim: 14336`, `num_heads: 24`. At construction time the model calls

```python
video_dit_config = apply_video_backbone_preset(
    dit_config=dict(video_dit_config),
    video_backbone_type="wan2_1_t2v",     # resolved from `video_backbone_type`
)
```

(`helpers/loader.py`), which **overwrites** the shape keys with the `Wan2.1-T2V-1.3B` preset. The
checkpoint confirms the real shape:

| Tensor | Shape in the checkpoint |
|---|---|
| `mixtures.video.patch_embedding.weight` | `(1536, 16, 1, 2, 2)` |
| `mixtures.video.blocks.0.self_attn.q.weight` | `(1536, 1536)` → 12 heads × 128 |
| `mixtures.video.blocks.0.ffn.0.weight` | `(8960, 1536)` |
| `mixtures.video.text_embedding.0.weight` | `(1536, 4096)` |
| `mixtures.video.head.head.weight` | `(64, 1536)` → 16 channels × patch(1,2,2) |
| backbone blocks | 30 (`blocks.0 … blocks.29`) |
| LoRA tensors | 600 = 30 layers × 10 target modules × (down, up) |

So: **Wan2.1-T2V-1.3B** (30 blocks, hidden 1536, ffn 8960, 12 heads), `in/out_dim = 48`,
`patch_size = [1, 2, 2]`, 16 latent channels.

## 3. Trainable / frozen split

`configure_trainable_modules()` freezes the backbone and keeps training:

* **LoRA** on all 30 blocks — targets `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.{0,2}`,
  rank **64**, alpha **128**, dropout 0.
* the **video head** (`head.head`, `head.modulation`), so the flow-matching objective still trains.
* the **temporal-group readout** + **proprio encoder**.

Measured on a trained checkpoint:

| Group | Parameters |
|---|---|
| backbone (frozen) | 1506.5 M |
| LoRA (trainable) | 87.5 M |
| video head (trainable) | 0.10 M |
| readout (trainable) | 33.4 M |
| proprio encoder (trainable) | 0.06 M |
| **trainable / total** | **121.0 M / 1539.9 M = 7.86 %** |

`wam_adapter.use_wam_adapter` is `false` on the production path (mid-layer
residual adapters are disabled); `freeze_backbone: true` and
`remove_original_action_expert: true` apply.

## 4. Data-side contract

| Item | Value |
|---|---|
| Cameras | 3 (`num_output_cameras: 3`), concatenated with `data.train.concat_multi_camera: robotwin` |
| RoboTwin resolution / frames | 384×320, 33 frames; factor-2 latents cached as `16 × 3 × 24 × 20` |
| Real-robot resolution / frames | raw 480×640 → 240×320, 33 frames, `action_dim = 14`, `proprio_dim = 14` |
| Cached latents | `scripts/precompute_wam_adapter_robotwin_full.sh` (idempotent, shape-validated) |
| Text | umT5 embeddings cached; `tokenizer_max_len: 128`, `load_text_encoder: false` |
| Proprio | appended as an extra context token (`_append_proprio_to_context`) and re-used as the readout's state token |
| Sampling | `global_sample_stride: 1`, `sample_start_stride: 1` (stride 1, no temporal subsampling) |
| Normalisation | `data.train.pretrained_norm_stats` → `dataset_stats.json` |
| Eval environment | `configs/sim_robotwin.yaml` (RoboTwin rollout config) |

## 5. Forward paths

Training (`training_loss` → `_training_loss_readout`):

1. `build_inputs` builds current-frame latents + `num_future_slots` future slots (τ = 1 noise).
2. **One** backbone pass over the whole current+future sequence; representations are captured from
   every layer in `readout_layer_indices`.
3. The cross-layer fusion splits the capture into temporal groups and tags each token with its
   physical action-step position (`_temporal_group_lengths`, `_prepare_temporal_group_capture`).
4. `action_decoder(...)` regresses the 32-step chunk (L2) from the fused contexts + action queries +
   proprio; the video head predicts the future latents from the same tokens.
5. Loss = video flow-matching loss + action L2 loss (optional `action_temporal_weighting`).

Inference (`infer_action`):

```python
model.infer_action(
    prompt, input_image, action_horizon, proprio=None, context=None,
    context_mask=None, num_video_frames=33,
)
```

The future slots stay at τ = 1 (never denoised) — a single conditioned pass, then the readout emits
the whole action chunk. `num_inference_steps` / `sigma_shift` are accepted but unused on this path.

### Temporal groups in detail

* `group_lengths` = `[tokens_per_frame] + [future_tokens / num_future_slots] × num_future_slots`, i.e.
  one **Current** group followed by equally sized **Early-Future** / **Late-Future** groups.
* `num_future_slots = 0` is the *current-frame-only* ablation: the fusion truncates its capture to
  the current-frame prefix (`current_only_tokens`) and every captured token sits at physical action
  time 0.
* Token positions are `frame_index × temporal_position_stride` (16 for RoboTwin), so a group keeps
  its meaning even when the latent grid is spatially downsampled by the factor-2 cache.
