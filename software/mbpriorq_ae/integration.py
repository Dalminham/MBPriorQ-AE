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


def _is_target(module: nn.Module, name: str) -> bool:
    is_linear = isinstance(module, nn.Linear) or "QuantLinear" in module.__class__.__name__
    return is_linear and any(marker in name for marker in TARGET_MARKERS)


def wrap_activation_linears(
    model: nn.Module,
    config: ActivationQuantizationConfig,
) -> list[str]:
    """Replace paper-scope Linear modules and return their fully qualified names."""
    replaced: list[str] = []

    def visit(module: nn.Module, prefix: str = "") -> None:
        if isinstance(module, nn.ModuleList):
            for index, child in enumerate(module):
                full_name = f"{prefix}[{index}]" if prefix else f"[{index}]"
                if _is_target(child, full_name):
                    module[index] = ActivationFakeQuantLinear(
                        child,
                        name=full_name,
                        config=config,
                    )
                    replaced.append(full_name)
                else:
                    visit(child, full_name)
            return
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _is_target(child, full_name):
                setattr(
                    module,
                    child_name,
                    ActivationFakeQuantLinear(child, name=full_name, config=config),
                )
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(model)
    if not replaced:
        raise RuntimeError("No paper-scope Linear modules were found")
    if "lm_head" not in replaced:
        raise RuntimeError("Paper-spec activation quantization requires lm_head")
    return replaced
