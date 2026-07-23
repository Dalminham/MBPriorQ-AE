from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from mbpriorq_ae.perplexity import (
    paper_compatible_by_layer_ppl,
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


class _ToyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, hidden_states, attention_mask=None):
        del attention_mask
        return (self.projection(hidden_states),)


class _ToyDecoder(nn.Module):
    def __init__(self, vocabulary, hidden_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocabulary, hidden_size)
        self.layers = nn.ModuleList([_ToyDecoderLayer(hidden_size) for _ in range(2)])
        self.norm = nn.Identity()


class _ToyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        vocabulary = 16
        hidden_size = 8
        self.model = _ToyDecoder(vocabulary, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocabulary, bias=False)
        self.config = SimpleNamespace(use_cache=False, hidden_size=hidden_size)

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=None)[0]
        hidden_states = self.model.norm(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


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

    def test_by_layer_matches_full_model_execution(self):
        torch.manual_seed(7)
        full_model = _ToyCausalLM()
        by_layer_model = _ToyCausalLM()
        by_layer_model.load_state_dict(full_model.state_dict())
        input_ids = (torch.arange(32) % 16).reshape(1, -1)

        full_ppl, full_windows, full_nll = paper_compatible_full_model_ppl(
            full_model,
            input_ids,
            sequence_length=8,
            device="cpu",
            progress=False,
        )
        by_layer_ppl, by_layer_windows, by_layer_nll = paper_compatible_by_layer_ppl(
            by_layer_model,
            input_ids,
            sequence_length=8,
            device="cpu",
            progress=False,
        )

        self.assertEqual(by_layer_windows, full_windows)
        self.assertAlmostEqual(by_layer_nll, full_nll, places=5)
        self.assertAlmostEqual(by_layer_ppl, full_ppl, places=6)


if __name__ == "__main__":
    unittest.main()
