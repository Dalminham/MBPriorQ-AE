from __future__ import annotations

import torch
import unittest

from mbpriorq_ae import GlobalEBW


class EbwTests(unittest.TestCase):
    FIELDS = {
        "weight_vmb_blocks": 0,
        "weight_total_blocks": 0,
        "activation_vmb_blocks": 0,
        "activation_total_blocks": 0,
        "kv_cache_vmb_blocks": 0,
        "kv_cache_total_blocks": 0,
    }

    def setUp(self):
        self.previous = {name: getattr(GlobalEBW, name) for name in self.FIELDS}
        self.previous_granularity = GlobalEBW.MBPRIORQ_REFINED_BLOCK_SIZE
        for name, value in self.FIELDS.items():
            setattr(GlobalEBW, name, value)

    def tearDown(self):
        for name, value in self.previous.items():
            setattr(GlobalEBW, name, value)
        GlobalEBW.configure_mbpriorq_refined_block_size(self.previous_granularity)

    def test_mask_and_scale_ebw_for_16_to_4_refinement(self):
        GlobalEBW.configure_mbpriorq_refined_block_size(4)
        mask = torch.tensor([[True, False, False, False]])
        GlobalEBW.record_vmb_mask("activation", mask)
        summary = GlobalEBW.summarize("activation")

        self.assertAlmostEqual(summary["vmb_partition"], 0.25)
        self.assertAlmostEqual(summary["mask_ebw"], 0.0625)
        self.assertAlmostEqual(summary["scale_ebw"], 0.5 + 1.5 * 0.25)
        self.assertAlmostEqual(summary["effective_ebw"], 4.0 + 0.0625 + 0.875)

    def test_refined_scale_ebw(self):
        for refined_block_size, vmb_scale_ebw in ((8, 1.0), (4, 2.0), (2, 4.0)):
            with self.subTest(refined_block_size=refined_block_size):
                GlobalEBW.configure_mbpriorq_refined_block_size(refined_block_size)
                self.assertAlmostEqual(
                    GlobalEBW.MBPRIORQ_VMB_SCALE_EBW, vmb_scale_ebw
                )


if __name__ == "__main__":
    unittest.main()
