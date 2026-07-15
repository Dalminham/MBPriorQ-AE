import pytest
import torch
import torch.nn as nn

from mbpriorq_ae.offload import _assign_module, quantize_state_weights


def test_assign_module_accepts_legacy_llama_rotary_buffer():
    module = nn.Linear(3, 2)
    state = {
        "weight": torch.ones_like(module.weight),
        "bias": torch.zeros_like(module.bias),
        "self_attn.rotary_emb.inv_freq": torch.ones(4),
    }

    assigned = _assign_module(module, state, "model.layers.0", torch.device("cpu"))

    assert torch.equal(assigned.weight, state["weight"])


def test_assign_module_still_rejects_unknown_checkpoint_keys():
    module = nn.Linear(3, 2)
    state = {
        "weight": torch.ones_like(module.weight),
        "bias": torch.zeros_like(module.bias),
        "unknown": torch.ones(1),
    }

    with pytest.raises(RuntimeError, match="unexpected=\\['unknown'\\]"):
        _assign_module(module, state, "model.layers.0", torch.device("cpu"))


def test_stacked_experts_keep_checkpoint_tensor_quantization_scope():
    class Quantizer:
        def __init__(self):
            self.names = []

        def fake_quantize_weight(self, tensor, name):
            self.names.append(name)
            return tensor.clone()

    quantizer = Quantizer()
    state = {"mlp.experts.gate_up_proj": torch.ones(3, 2, 16)}

    count = quantize_state_weights(state, "model.layers.0", quantizer)

    assert count == 1
    assert quantizer.names == ["model.layers.0.mlp.experts.gate_up_proj"]
