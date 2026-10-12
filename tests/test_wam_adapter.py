"""Tests for the canonical temporal-group readout."""

import unittest
from unittest import mock

import torch

from wam_adapter.models.wan22.wam_adapter_readout import (
    TemporalGroupCrossLayerFusion,
)
from wam_adapter.models.wan22.wam_adapter import (
    WAMAdapter,
)


def _install_shell():
    model = WAMAdapter.__new__(WAMAdapter)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.remove_original_action_expert = True
    model.use_wam_adapter = False
    model.proprio_encoder = None
    model.vae = mock.Mock(temporal_downsample_factor=4)
    video_expert = mock.Mock()
    video_expert.hidden_dim = 48
    video_expert.num_heads = 6
    video_expert.blocks = [mock.Mock() for _ in range(4)]
    video_expert.action_conditioned = False
    model.video_expert = video_expert
    old_expert = mock.Mock(action_dim=7)
    model.readout = old_expert
    return model


class TemporalGroupReadoutTest(unittest.TestCase):
    def test_production_model_is_a_direct_torch_module(self):
        self.assertEqual(WAMAdapter.__bases__, (torch.nn.Module,))
        self.assertIn("training_loss", WAMAdapter.__dict__)
        self.assertIn("infer_action", WAMAdapter.__dict__)
        self.assertIn("save_checkpoint", WAMAdapter.__dict__)
        self.assertIn("load_checkpoint", WAMAdapter.__dict__)

    def test_install_uses_independent_block_head_group_layer_weights(self):
        model = _install_shell()
        model._install_readout(
            {
                "action_hidden_dim": 32,
                "action_num_heads": 4,
                "action_ffn_dim": 64,
                "action_decoder_layers": 2,
                "readout_layer_indices": [0, 1, 2, 3],
                "num_future_slots": 2,
                "temporal_position_stride": 16,
            }
        )
        fusion = model.action_decoder.fusion
        self.assertIsInstance(fusion, TemporalGroupCrossLayerFusion)
        self.assertEqual(fusion.capture_basis, "canonical")
        self.assertFalse(fusion.requires_rope_freqs)
        self.assertEqual(tuple(fusion.layer_weights.shape), (2, 4, 3, 4))
        self.assertEqual(model.action_decoder_config["group_fusion"],
                         "independent_unconstrained")

    def test_physical_positions_preserve_every_latent_slot(self):
        model = _install_shell()
        model._install_readout(
            {
                "action_hidden_dim": 32,
                "action_num_heads": 4,
                "action_ffn_dim": 64,
                "readout_layer_indices": [0, 1],
                "temporal_position_stride": 16,
            }
        )
        pre = {"meta": {"tokens_per_frame": 2, "grid_size": (3, 1, 2)}}
        lengths = (2, 2, 2)
        positions = model._prepare_temporal_group_capture(pre, lengths)
        self.assertEqual(
            positions.tolist(),
            [0, 0, 16, 16, 32, 32],
        )

    def test_missing_physical_stride_fails_loudly(self):
        model = _install_shell()
        model._install_readout(
            {
                "action_hidden_dim": 32,
                "action_num_heads": 4,
                "action_ffn_dim": 64,
                "readout_layer_indices": [0, 1],
            }
        )
        pre = {"meta": {"tokens_per_frame": 2, "grid_size": (3, 1, 2)}}
        with self.assertRaisesRegex(ValueError, "temporal_position_stride"):
            model._prepare_temporal_group_capture(pre, (2, 2, 2))

    def test_group_specific_unconstrained_weights_fuse_online(self):
        fusion = TemporalGroupCrossLayerFusion(
            video_hidden_dim=4,
            video_num_heads=1,
            action_hidden_dim=4,
            action_num_heads=1,
            num_action_blocks=1,
            num_temporal_groups=3,
            selected_layers=[0, 1],
        )
        with torch.no_grad():
            fusion.key_projection.weight.copy_(torch.eye(4))
            fusion.key_projection.bias.zero_()
            fusion.value_projection.weight.copy_(torch.eye(4))
            fusion.value_projection.bias.zero_()
            # Each group selects a different unconstrained combination.
            fusion.layer_weights.copy_(
                torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]]])
            )
        fusion.prepare_group_capture((1, 1, 1))
        fusion.begin_capture()
        layer0 = torch.ones(1, 3, 4)
        layer1 = torch.full((1, 3, 4), 3.0)
        fusion.accumulate(0, layer0, layer0)
        fusion.accumulate(1, layer1, layer1)
        # Accumulation already contains the desired current/early/late mixtures:
        # 1, 3, and 2*1-3=-1.  No full per-layer cache is retained.
        self.assertTrue(torch.equal(fusion._value_sum[0, 0, 0, :, 0],
                                    torch.tensor([1.0, 3.0, -1.0])))
        self.assertFalse(hasattr(fusion, "captured_layers"))
        groups = fusion.finish_capture_groups((1, 1, 1), [0.0, 4.0, 8.0])
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0][0].value.shape, (1, 1, 1, 4))

    def test_shared_temporal_fusion_broadcasts_one_layer_mixture(self):
        fusion = TemporalGroupCrossLayerFusion(
            video_hidden_dim=4,
            video_num_heads=1,
            action_hidden_dim=4,
            action_num_heads=1,
            num_action_blocks=1,
            num_temporal_groups=3,
            selected_layers=[0, 1],
            share_temporal_group_layer_weights=True,
        )
        self.assertEqual(tuple(fusion.layer_weights.shape), (1, 1, 1, 2))
        with torch.no_grad():
            fusion.key_projection.weight.copy_(torch.eye(4))
            fusion.key_projection.bias.zero_()
            fusion.value_projection.weight.copy_(torch.eye(4))
            fusion.value_projection.bias.zero_()
            fusion.layer_weights.copy_(torch.tensor([[[[2.0, -1.0]]]]))
        fusion.prepare_group_capture((1, 1, 1))
        fusion.begin_capture()
        layer0 = torch.tensor([[[1.0] * 4, [2.0] * 4, [3.0] * 4]])
        layer1 = torch.tensor([[[4.0] * 4, [5.0] * 4, [6.0] * 4]])
        fusion.accumulate(0, layer0, layer0)
        fusion.accumulate(1, layer1, layer1)
        self.assertTrue(
            torch.equal(
                fusion._value_sum[0, 0, 0, :, 0],
                torch.tensor([-2.0, -1.0, 0.0]),
            )
        )

    def test_model_config_selects_shared_temporal_fusion(self):
        model = _install_shell()
        model._install_readout(
            {
                "action_hidden_dim": 32,
                "action_num_heads": 4,
                "action_ffn_dim": 64,
                "action_decoder_layers": 2,
                "readout_layer_indices": [0, 1, 2, 3],
                "num_future_slots": 2,
                "temporal_position_stride": 16,
                "share_temporal_group_layer_weights": True,
            }
        )
        fusion = model.action_decoder.fusion
        self.assertEqual(tuple(fusion.layer_weights.shape), (2, 4, 1, 4))
        self.assertEqual(
            model.action_decoder_config["group_fusion"],
            "shared_unconstrained",
        )

    def test_shared_temporal_fusion_checkpoint_reconstructs_strictly(self):
        config = {
            "action_hidden_dim": 32,
            "action_num_heads": 4,
            "action_ffn_dim": 64,
            "action_decoder_layers": 2,
            "readout_layer_indices": [0, 1, 2, 3],
            "num_future_slots": 2,
            "temporal_position_stride": 16,
            "share_temporal_group_layer_weights": True,
        }
        trained = _install_shell()
        trained._install_readout(config)
        checkpoint_config = dict(trained.action_decoder_config)
        checkpoint_state = trained.action_decoder.state_dict()

        evaluated = _install_shell()
        evaluated._install_readout(checkpoint_config)
        evaluated.action_decoder.load_state_dict(checkpoint_state, strict=True)
        self.assertEqual(
            tuple(evaluated.action_decoder.fusion.layer_weights.shape),
            (2, 4, 1, 4),
        )

    def test_future_and_action_gradients_reach_group_weights(self):
        fusion = TemporalGroupCrossLayerFusion(
            video_hidden_dim=8,
            video_num_heads=2,
            action_hidden_dim=8,
            action_num_heads=2,
            num_action_blocks=2,
            num_temporal_groups=3,
            selected_layers=[0, 1],
        )
        fusion.prepare_group_capture((2, 2, 2))
        fusion.begin_capture()
        for layer in (0, 1):
            key = torch.randn(1, 6, 8, requires_grad=True)
            value = torch.randn(1, 6, 8, requires_grad=True)
            fusion.accumulate(layer, key, value)
        groups = fusion.finish_capture_groups(
            (2, 2, 2), torch.tensor([0, 0, 4, 4, 8, 8])
        )
        loss = sum(group[0].key.square().mean() + group[1].value.square().mean()
                   for group in groups)
        loss.backward()
        self.assertIsNotNone(fusion.layer_weights.grad)
        self.assertGreater(float(fusion.layer_weights.grad.abs().sum()), 0.0)

    def test_no_future_slots_installs_single_current_group(self):
        model = _install_shell()
        model._install_readout(
            {
                "action_hidden_dim": 32,
                "action_num_heads": 4,
                "action_ffn_dim": 64,
                "action_decoder_layers": 2,
                "readout_layer_indices": [0, 1, 2, 3],
                "num_future_slots": 0,
                "temporal_position_stride": 16,
            }
        )
        fusion = model.action_decoder.fusion
        self.assertEqual(tuple(fusion.layer_weights.shape), (2, 4, 1, 4))
        self.assertEqual(model.action_decoder_config["num_future_slots"], 0)
        pre = {"meta": {"tokens_per_frame": 2, "grid_size": (3, 1, 2)}}
        lengths = model._temporal_group_lengths(pre)
        self.assertEqual(lengths, (2,))
        positions = model._prepare_temporal_group_capture(pre, lengths)
        self.assertEqual(positions.tolist(), [0.0, 0.0])
        self.assertEqual(fusion.current_only_tokens, 2)

    def test_readout_current_only_capture_truncates_future_tokens(self):
        fusion = TemporalGroupCrossLayerFusion(
            video_hidden_dim=8,
            video_num_heads=2,
            action_hidden_dim=8,
            action_num_heads=2,
            num_action_blocks=2,
            num_temporal_groups=1,
            selected_layers=[0, 1],
        )
        # A 6-token joint sequence (2 current + 4 future) is fed to the fusion,
        # but current-only capture must keep only the 2 current-frame tokens.
        fusion.current_only_tokens = 2
        fusion.prepare_group_capture((2,))
        for layer in (0, 1):
            fusion.accumulate(layer, torch.randn(1, 6, 8), torch.randn(1, 6, 8))
        groups = fusion.finish_capture_groups((2,), [0.0, 0.0])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        for block in groups[0]:
            self.assertEqual(block.key.shape[2], 2)
            self.assertEqual(block.value.shape[2], 2)


if __name__ == "__main__":
    unittest.main()
