from __future__ import annotations

import math
import unittest

import torch

from mbpriorq_ae import MBPriorQ_Quantizer


def quantizer_args(**overrides):
    args = {
        "name": "ae.test",
        "device": "cpu",
        "quant_bit": 4,
        "quant_sym": True,
        "model_type": "cloud",
        "ablation_mode": "paper",
        "random_seed": 20260606,
        "refined_block_size": 4,
        "using_imatrix": False,
        "imatrix_file_name": "unused",
        "vmb_profile_enable": False,
        "metadata_target": "activation",
    }
    args.update(overrides)
    return args


class QuantizationTests(unittest.TestCase):
    def test_e2m1_tie_rounding_and_sign_codes(self):
        quantizer = MBPriorQ_Quantizer(quantizer_args())
        values = torch.tensor(
            [-6.0, -5.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25,
             0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
        )
        expected = torch.tensor(
            [15, 14, 14, 12, 12, 10, 10, 8, 0, 0, 2, 2, 4, 4, 6, 6, 7],
            dtype=torch.uint8,
        )
        self.assertTrue(torch.equal(quantizer._cast_fp4(values), expected))

    def test_regular_path_round_trip_is_finite_and_shape_preserving(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                quantizer = MBPriorQ_Quantizer(quantizer_args())
                data = torch.linspace(-7.0, 7.0, 128, dtype=torch.float32).reshape(4, 32).to(dtype)
                global_scale = data.abs().amax().float() / (6 * torch.finfo(torch.float8_e4m3fn).max)
                packed, block_scale = quantizer._quantize_nvfp4(data, 16, global_scale)
                result = quantizer._dequantize_nvfp4(
                    packed, block_scale, global_scale, data.shape, dtype
                )
                self.assertEqual(result.shape, data.shape)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.isfinite(result.float()).all())

    def test_mbpriorq_paper_path_handles_calibration_and_prior(self):
        for refined_block_size in (2, 4, 8):
            with self.subTest(refined_block_size=refined_block_size):
                torch.manual_seed(1308)
                quantizer = MBPriorQ_Quantizer(
                    quantizer_args(refined_block_size=refined_block_size)
                )
                calibration = torch.randn(8, 64)
                later = torch.randn(8, 64)

                first = quantizer.fake_quantize_activation(
                    calibration,
                    name="model.layers.0.mlp.down_proj",
                    tensor_shape=calibration.shape,
                )
                second = quantizer.fake_quantize_activation(
                    later,
                    name="model.layers.0.mlp.down_proj",
                    tensor_shape=later.shape,
                )

                self.assertEqual(first.shape, calibration.shape)
                self.assertEqual(second.shape, later.shape)
                self.assertTrue(torch.isfinite(first).all())
                self.assertTrue(torch.isfinite(second).all())
                self.assertTrue(quantizer.mask_set)

    def test_random_same_ratio_is_deterministic_for_fixed_seed(self):
        torch.manual_seed(1308)
        calibration = torch.randn(8, 64)
        later = torch.randn(8, 64)

        outputs = []
        for _ in range(2):
            quantizer = MBPriorQ_Quantizer(
                quantizer_args(ablation_mode="random_same_ratio", random_seed=7)
            )
            quantizer.fake_quantize_activation(
                calibration, name="layer", tensor_shape=calibration.shape
            )
            outputs.append(
                quantizer.fake_quantize_activation(
                    later, name="layer", tensor_shape=later.shape
                )
            )

        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_invalid_refined_granularity_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "refined_block_size"):
            MBPriorQ_Quantizer(quantizer_args(refined_block_size=16))

    def test_zero_tensor_does_not_create_non_finite_activation_values(self):
        quantizer = MBPriorQ_Quantizer(
            quantizer_args()
        )
        data = torch.zeros(4, 32)
        result = quantizer.fake_quantize_activation(
            data, name="zero", tensor_shape=data.shape
        )
        self.assertEqual(result.shape, data.shape)
        self.assertTrue(math.isfinite(float(result.float().sum())))

    def test_cloud_prior_accepts_single_token_decode_step(self):
        quantizer = MBPriorQ_Quantizer(quantizer_args())
        calibration = torch.randn(1, 8, 32)
        decode_step = torch.randn(1, 1, 32)
        quantizer.fake_quantize_activation(
            calibration,
            name="model.layers.0.self_attn.q_proj",
            tensor_shape=(32, 32),
        )
        result = quantizer.fake_quantize_activation(
            decode_step,
            name="model.layers.0.self_attn.q_proj",
            tensor_shape=(32, 32),
        )
        self.assertEqual(result.shape, decode_step.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_feature_modes_preserve_standard_deviation_selection_count(self):
        torch.manual_seed(1308)
        data = torch.randn(8, 32)
        name = "model.layers[0].mlp.up_proj"
        gradient = {"model.layers.0.mlp.up_proj": torch.rand(32)}
        reference = MBPriorQ_Quantizer(quantizer_args())
        std = reference._feature_std_metric(data)
        threshold, _, _ = reference._search_threshold_for_replacement(data, name)
        expected_count = int((std > threshold).sum().item())

        for feature in ("std", "diff", "grad", "diff_grad", "std_grad"):
            with self.subTest(feature=feature):
                quantizer = MBPriorQ_Quantizer(
                    quantizer_args(
                        feature_mode=feature,
                        gradient_info=gradient if "grad" in feature else None,
                    )
                )
                selected, _ = quantizer._select_feature_mask(data, name, data.dtype)
                self.assertEqual(int(selected.sum().item()), expected_count)

    def test_gradient_feature_requires_matching_calibration_entry(self):
        quantizer = MBPriorQ_Quantizer(
            quantizer_args(feature_mode="grad", gradient_info={"other": torch.ones(32)})
        )
        with self.assertRaisesRegex(KeyError, "Gradient calibration"):
            quantizer.fake_quantize_activation(
                torch.randn(4, 32),
                name="model.layers[0].mlp.up_proj",
                tensor_shape=(16, 32),
            )


if __name__ == "__main__":
    unittest.main()
