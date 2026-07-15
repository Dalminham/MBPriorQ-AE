"""Minimal Hugging Face integration for MBPriorQ artifact evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mbpriorq import MBPriorQ_Quantizer


TARGET_MARKERS = ("self_attn", "mlp", "lm_head", "block_sparse_moe")


@dataclass(frozen=True)
class ActivationQuantizationConfig:
    method: str
    model_type: str = "cloud"
    ablation_mode: str = "paper"
    random_seed: int = 20260606
    refined_block_size: int = 4


def quantizer_arguments(
    *,
    name: str | None,
    method: str,
    model_type: str = "cloud",
    ablation_mode: str = "paper",
    random_seed: int = 20260606,
    refined_block_size: int = 4,
    using_imatrix: bool = False,
    imatrix_file_name: str | None = None,
) -> dict:
    """Build the exact argument surface used by the curated quantizers."""
    if method != "mbpriorq":
        raise ValueError(f"Unsupported quantization method: {method}")
    return {
        "name": name,
        "device": "cpu",
        "quant_bit": 4,
        "quant_sym": True,
        "model_type": model_type,
        "ablation_mode": ablation_mode,
        "random_seed": int(random_seed),
        "refined_block_size": int(refined_block_size),
        "using_imatrix": bool(using_imatrix),
        "imatrix_file_name": imatrix_file_name,
        "vmb_profile_enable": False,
        "metadata_target": "activation",
    }


def make_quantizer(*, method: str, name: str | None, **kwargs):
    args = quantizer_arguments(name=name, method=method, **kwargs)
    return MBPriorQ_Quantizer(args=args, weight=None)


class ActivationFakeQuantLinear(nn.Module):
    """Linear layer with source-faithful activation fake quantization.

    The original EasyLLM wrapper overwrites the Linear input with its
    fake-quantized/dequantized value immediately before ``F.linear``. This
    compact wrapper preserves that behavior without importing unrelated
    quantization backends.
    """

    def __init__(
        self,
        original: nn.Module,
        *,
        name: str,
        config: ActivationQuantizationConfig,
    ) -> None:
        super().__init__()
        if not hasattr(original, "weight"):
            raise TypeError(f"Layer {name!r} has no weight parameter")
        self.in_features = getattr(original, "in_features", original.weight.shape[-1])
        self.out_features = getattr(original, "out_features", original.weight.shape[-2])
        self.weight = original.weight
        self.bias = getattr(original, "bias", None)
        self.tensor_name = name
        self.quantizer = make_quantizer(
            method=config.method,
            name=name,
            model_type=config.model_type,
            ablation_mode=config.ablation_mode,
            random_seed=config.random_seed,
            refined_block_size=config.refined_block_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.fake_quantize_activation(
            x,
            self.tensor_name,
            self.weight.shape,
        )
        x.copy_(quantized)
        return F.linear(x, self.weight, self.bias)


class ActivationFakeQuantModule(nn.Module):
    """Quantize a module input while preserving the module's own forward path.

    This form is used by generation backends that may attach device-dispatch
    hooks to the original module. It deliberately delegates computation to the
    wrapped module instead of reconstructing a Linear operation.
    """

    def __init__(
        self,
        original: nn.Module,
        *,
        name: str,
        config: ActivationQuantizationConfig,
    ) -> None:
        super().__init__()
        if not hasattr(original, "weight"):
            raise TypeError(f"Layer {name!r} has no weight parameter")
        self.original = original
        self.tensor_name = name
        self.quantizer = make_quantizer(
            method=config.method,
            name=name,
            model_type=config.model_type,
            ablation_mode=config.ablation_mode,
            random_seed=config.random_seed,
            refined_block_size=config.refined_block_size,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        quantized = self.quantizer.fake_quantize_activation(
            x,
            self.tensor_name,
            self.original.weight.shape,
        )
        return self.original(quantized, *args, **kwargs)


class ActivationFakeQuantQwen3VLExperts(nn.Module):
    """Activation fake quantization for Qwen3-VL's stacked text experts."""

    def __init__(
        self,
        original: nn.Module,
        *,
        name: str,
        config: ActivationQuantizationConfig,
    ) -> None:
        super().__init__()
        self.original = original
        self.tensor_name = name
        self.gate_up_quantizer = make_quantizer(
            method=config.method,
            name=f"{name}.gate_up_proj",
            model_type=config.model_type,
            ablation_mode=config.ablation_mode,
            random_seed=config.random_seed,
            refined_block_size=config.refined_block_size,
        )
        self.down_quantizer = make_quantizer(
            method=config.method,
            name=f"{name}.down_proj",
            model_type=config.model_type,
            ablation_mode=config.ablation_mode,
            random_seed=config.random_seed,
            refined_block_size=config.refined_block_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        original = self.original
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, original.hidden_size)
        next_states = torch.zeros_like(hidden_states)
        gate_up_shape = (2 * original.expert_dim, original.hidden_size)
        down_shape = (original.hidden_size, original.expert_dim)

        # Only materialize tokens actually routed to an expert. Repeating all
        # tokens for every expert is unnecessary and exceeds edge-device memory.
        for expert_tensor in torch.unique(router_indices):
            expert = int(expert_tensor.item())
            token_indices = torch.nonzero(
                router_indices.eq(expert).any(dim=-1), as_tuple=False
            ).flatten()
            if token_indices.numel() == 0:
                continue
            current = hidden_states.index_select(0, token_indices)
            current = self.gate_up_quantizer.fake_quantize_activation(
                current,
                f"{self.tensor_name}.gate_up_proj",
                gate_up_shape,
            )
            gate_up = current @ original.gate_up_proj[expert]
            gate, up = gate_up.chunk(2, dim=-1)
            current = up * original.act_fn(gate)
            current = self.down_quantizer.fake_quantize_activation(
                current,
                f"{self.tensor_name}.down_proj",
                down_shape,
            )
            output = current @ original.down_proj[expert]
            output *= routing_weights[token_indices, expert].unsqueeze(-1)
            next_states.index_add_(0, token_indices, output.to(next_states.dtype))
        return next_states.view(batch_size, -1, original.hidden_size)


def _is_target(module: nn.Module, name: str) -> bool:
    is_linear = isinstance(module, nn.Linear) or "QuantLinear" in module.__class__.__name__
    return is_linear and any(marker in name for marker in TARGET_MARKERS)


def wrap_activation_linears(
    model: nn.Module,
    config: ActivationQuantizationConfig,
    *,
    prefix: str = "",
    require_lm_head: bool = True,
    preserve_module_forward: bool = False,
) -> list[str]:
    """Replace paper-scope Linear modules and return their fully qualified names."""
    replaced: list[str] = []

    wrapper = ActivationFakeQuantModule if preserve_module_forward else ActivationFakeQuantLinear

    def visit(module: nn.Module, current_prefix: str = "") -> None:
        if isinstance(module, nn.ModuleList):
            for index, child in enumerate(module):
                full_name = (
                    f"{current_prefix}[{index}]" if current_prefix else f"[{index}]"
                )
                if _is_target(child, full_name):
                    module[index] = wrapper(
                        child,
                        name=full_name,
                        config=config,
                    )
                    replaced.append(full_name)
                else:
                    visit(child, full_name)
            return
        for child_name, child in list(module.named_children()):
            full_name = (
                f"{current_prefix}.{child_name}" if current_prefix else child_name
            )
            if child.__class__.__name__ == "Qwen3VLMoeTextExperts":
                setattr(
                    module,
                    child_name,
                    ActivationFakeQuantQwen3VLExperts(
                        child,
                        name=full_name,
                        config=config,
                    ),
                )
                replaced.append(full_name)
                continue
            if _is_target(child, full_name):
                setattr(
                    module,
                    child_name,
                    wrapper(child, name=full_name, config=config),
                )
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model, prefix)
    if not replaced:
        raise RuntimeError("No paper-scope Linear modules were found")
    if require_lm_head and "lm_head" not in replaced:
        raise RuntimeError("Paper-spec activation quantization requires lm_head")
    return replaced
