from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mbpriorq_ae import GlobalEBW
from mbpriorq_ae.kv_cache import build_kv_cache


class KVCacheTests(unittest.TestCase):
    def setUp(self):
        GlobalEBW.reset()
        torch.manual_seed(1308)
        self.config = SimpleNamespace(num_hidden_layers=2)

    def test_nvfp4_k_sequence_v_head_dim_preserves_shape(self):
        cache = build_kv_cache(self.config, "nvfp4")
        key = torch.randn(1, 2, 17, 32, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        key_out, value_out = cache.update(key, value, 0)
        self.assertEqual(key_out.shape, key.shape)
        self.assertEqual(value_out.shape, value.shape)
        self.assertTrue(torch.isfinite(key_out.float()).all())
        self.assertTrue(torch.isfinite(value_out.float()).all())

    def test_mbpriorq_records_kv_cache_metadata(self):
        cache = build_kv_cache(self.config, "mbpriorq")
        key = torch.randn(1, 2, 16, 32, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        cache.update(key, value, 0)
        summary = GlobalEBW.summarize("kv_cache")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["mask_ebw"], 0.0625)
        self.assertGreaterEqual(summary["effective_ebw"], 4.5625)


if __name__ == "__main__":
    unittest.main()
