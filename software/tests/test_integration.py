import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from mbpriorq_ae import GlobalEBW
from mbpriorq_ae.checkpoint import stream_fake_quantize_checkpoint
from mbpriorq_ae.integration import (
    ActivationFakeQuantLinear,
    ActivationQuantizationConfig,
    make_quantizer,
)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        GlobalEBW.reset()
        torch.manual_seed(17)

    def test_activation_wrapper_matches_direct_quantizer(self):
        linear = nn.Linear(32, 16, bias=True)
        wrapped = ActivationFakeQuantLinear(
            linear,
            name="model.layers.0.mlp.up_proj",
            config=ActivationQuantizationConfig(method="mbpriorq"),
        )
        direct = make_quantizer(
            method="mbpriorq",
            name="model.layers.0.mlp.up_proj",
            model_type="cloud",
            ablation_mode="paper",
            refined_block_size=4,
        )
        source = torch.randn(1, 8, 32)
        expected_input = direct.fake_quantize_activation(
            source.clone(),
            "model.layers.0.mlp.up_proj",
            linear.weight.shape,
        )
        expected = torch.nn.functional.linear(expected_input, linear.weight, linear.bias)
        actual = wrapped(source.clone())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streamed_checkpoint_preserves_tied_weight_contract(self):
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("safetensors is unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            tensors = {
                "lm_head.weight": torch.randn(32, 16),
                "model.embed_tokens.weight": torch.randn(32, 16),
                "model.layers.0.mlp.up_proj.weight": torch.randn(16, 16),
                "model.layers.0.input_layernorm.weight": torch.randn(16),
            }
            save_file(tensors, source / "model.safetensors", metadata={"format": "pt"})
            (source / "config.json").write_text(
                '{"tie_word_embeddings": true}', encoding="utf-8"
            )

            stats = stream_fake_quantize_checkpoint(
                source_path=source,
                output_path=output,
                method="mbpriorq",
            )
            self.assertTrue(stats["lm_head_quantized"])
            self.assertEqual(stats["lm_head_saved_as"], "model.embed_tokens.weight")
            with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
            self.assertIn("model.embed_tokens.weight", keys)
            self.assertNotIn("lm_head.weight", keys)
            self.assertTrue((output / "mbpriorq_ae_prequant_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
