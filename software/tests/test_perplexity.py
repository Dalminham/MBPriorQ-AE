from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from mbpriorq_ae.perplexity import (
    paper_compatible_full_model_ppl,
    paper_compatible_kv_cache_ppl,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(use_cache=False)
        self.batch_shapes = []

    def forward(self, input_ids, **kwargs):
        self.batch_shapes.append(tuple(input_ids.shape))
        vocabulary = 16
        logits = torch.zeros(
            (*input_ids.shape, vocabulary), dtype=torch.float32, device=input_ids.device
        )
        logits.scatter_(-1, input_ids.unsqueeze(-1), 1.0)
        return SimpleNamespace(logits=logits)


class PerplexityTests(unittest.TestCase):
    def test_full_model_groups_true_batches(self):
        model = _ToyModel()
        input_ids = (torch.arange(32) % 16).reshape(1, -1)
        _, windows, _ = paper_compatible_full_model_ppl(
            model,
            input_ids,
            sequence_length=8,
            batch_size=2,
            device="cpu",
            progress=False,
        )
        self.assertEqual(windows, 4)
        self.assertEqual(model.batch_shapes, [(2, 8), (2, 8)])

    def test_kv_cache_protocol_keeps_final_partial_chunk(self):
        model = _ToyModel()
        input_ids = (torch.arange(18) % 16).reshape(1, -1)
        _, chunks, _ = paper_compatible_kv_cache_ppl(
            model,
            input_ids,
            sequence_length=8,
            device="cpu",
            progress=False,
        )
        self.assertEqual(chunks, 3)
        self.assertEqual(model.batch_shapes, [(1, 8), (1, 8), (1, 2)])


if __name__ == "__main__":
    unittest.main()
