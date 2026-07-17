from __future__ import annotations

import unittest

import torch

from mbpriorq_ae import GlobalEBW, MBPriorQ_Quantizer


class SmokeTests(unittest.TestCase):
    def test_deterministic_prior_and_ebw_path(self):
        torch.manual_seed(1308)
        calibration = torch.randn(8, 64)
        later = torch.randn(8, 64)
        GlobalEBW.reset()
        quantizer = MBPriorQ_Quantizer(
            {
                "name": "ae.smoke",
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
        )

        calibration_out = quantizer.fake_quantize_activation(
            calibration, name="layer", tensor_shape=calibration.shape
        )
        prior_out = quantizer.fake_quantize_activation(
            later, name="layer", tensor_shape=later.shape
        )
        summary = GlobalEBW.summarize("activation")

        self.assertAlmostEqual(float(calibration_out.sum()), 18.2597713470459)
        self.assertAlmostEqual(float(prior_out.sum()), -25.553970336914062)
        self.assertEqual(summary["vmb_blocks"], 34)
        self.assertEqual(summary["total_blocks"], 64)
        self.assertAlmostEqual(summary["mask_ebw"], 0.0625)
        self.assertAlmostEqual(summary["effective_ebw"], 5.359375)


if __name__ == "__main__":
    unittest.main()
