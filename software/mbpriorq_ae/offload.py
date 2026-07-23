"""Minimal safetensors layer streaming for MBPriorQ artifact evaluation.

The runtime keeps hidden states in CPU DRAM and materializes one decoder layer
at a time on the GPU. It is intentionally a correctness-oriented AE path, not
the separate EasyOffload research runtime.
"""

from __future__ import annotations

import gc
import inspect
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .integration import (
    ActivationFakeQuantLinear,
    ActivationQuantizationConfig,
    wrap_activation_linears,
)


INDEX_NAME = "model.safetensors.index.json"
WEIGHT_MARKERS = (".self_attn.", ".mlp.", ".block_sparse_moe.")


@dataclass
class ExecutionPlan:
    family: str
    forward_model: nn.Module
    base_model: nn.Module
    lm_head: nn.Module
    config: object
    hidden_size: int
    embed_prefix: str
    layer_prefix_template: str
    norm_prefix: str
    lm_head_prefix: str
    tied_lm_head_prefix: str


@dataclass(frozen=True)
class StreamedPPLResult:
    perplexity: float
    windows: int
    total_nll: float
    wrapped_linear_count: int
    quantized_weight_count: int
    model_family: str


@dataclass(frozen=True)
class StreamedLayerSmokeResult:
    requested_layers: int
    completed_layers: int
    total_layers: int
    wrapped_linear_count: int
    quantized_weight_count: int
    model_family: str
    layers: tuple[dict, ...]


class _CaptureInterrupt(Exception):
    pass


def _safe_open():
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Streamed PPL requires safetensors") from error
    return safe_open


def load_safetensors_index(checkpoint: str | Path) -> dict:
    checkpoint = Path(checkpoint)
    index_path = checkpoint / INDEX_NAME
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if "weight_map" not in index:
            raise ValueError(f"Missing weight_map in {index_path}")
        return index
    single = checkpoint / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(
            f"Expected {INDEX_NAME} or model.safetensors under {checkpoint}"
        )
    safe_open = _safe_open()
    with safe_open(str(single), framework="pt", device="cpu") as handle:
        weight_map = {name: single.name for name in handle.keys()}
    return {"metadata": {"total_size": single.stat().st_size}, "weight_map": weight_map}


def grouped_names_for_prefix(weight_map: dict[str, str], prefix: str):
    grouped: dict[str, list[str]] = defaultdict(list)
    prefix_dot = prefix + "."
    for tensor_name, shard_name in weight_map.items():
        if tensor_name == prefix or tensor_name.startswith(prefix_dot):
            grouped[shard_name].append(tensor_name)
    return grouped


def load_state_for_prefix(
    checkpoint: str | Path,
    weight_map: dict[str, str],
    prefix: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    grouped = grouped_names_for_prefix(weight_map, prefix)
    if not grouped:
        raise KeyError(f"No checkpoint tensors found for {prefix!r}")
    checkpoint = Path(checkpoint)
    safe_open = _safe_open()
    prefix_dot = prefix + "."
    state: dict[str, torch.Tensor] = {}
    for shard_name, tensor_names in grouped.items():
        with safe_open(str(checkpoint / shard_name), framework="pt", device="cpu") as handle:
            for tensor_name in tensor_names:
                key = tensor_name[len(prefix_dot) :]
                tensor = handle.get_tensor(tensor_name)
                if tensor.is_floating_point() and tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                state[key] = tensor
    return state


def _weight_target(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2:
        return False
    if name == "lm_head.weight":
        return True
    regular_weight = name.endswith(".weight")
    stacked_expert = ".mlp.experts." in name
    return (regular_weight or stacked_expert) and any(
        marker in name for marker in WEIGHT_MARKERS
    )


def _module_name(parameter_name: str) -> str:
    if parameter_name == "lm_head.weight":
        return "lm_head"
    if parameter_name.endswith(".weight"):
        return parameter_name[: -len(".weight")]
    return parameter_name


@torch.no_grad()
def quantize_state_weights(
    state: dict[str, torch.Tensor],
    prefix: str,
    quantizer,
) -> int:
    count = 0
    for local_name, tensor in list(state.items()):
        full_name = f"{prefix}.{local_name}"
        if not _weight_target(full_name, tensor):
            continue
        quantized = quantizer.fake_quantize_weight(tensor, _module_name(full_name))
        if quantized.shape != tensor.shape or quantized.dtype != tensor.dtype:
            raise RuntimeError(f"Weight quantization changed the contract for {full_name}")
        state[local_name] = quantized.contiguous()
        count += 1
    return count


@torch.no_grad()
def quantize_model_weights(model: nn.Module, quantizer) -> int:
    """Apply weight fake quantization to an in-memory model."""
    count = 0
    seen: set[int] = set()

    # Quantize lm_head first so tied embeddings follow the paper's lm_head rule.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        parameter = lm_head.weight
        quantized = quantizer.fake_quantize_weight(parameter.data, "lm_head")
        parameter.data.copy_(quantized)
        seen.add(id(parameter))
        count += 1

    for name, parameter in model.named_parameters():
        if id(parameter) in seen or not _weight_target(name, parameter):
            continue
        quantized = quantizer.fake_quantize_weight(parameter.data, _module_name(name))
        parameter.data.copy_(quantized)
        seen.add(id(parameter))
        count += 1
    if count == 0:
        raise RuntimeError("No paper-scope weights were quantized")
    return count


def _assign_module(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    prefix: str,
    device: torch.device,
) -> nn.Module:
    expected = set(module.state_dict())
    compatibility_buffers = {
        key
        for key in state
        if key == "self_attn.rotary_emb.inv_freq" and key not in expected
    }
    filtered_state = {
        key: value for key, value in state.items() if key not in compatibility_buffers
    }
    missing, unexpected = module.load_state_dict(
        filtered_state, strict=False, assign=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"load_state_dict({prefix}) missing={missing}, unexpected={unexpected}"
        )
    return module.to(device)


def unload_to_meta(module: nn.Module) -> None:
    module.to_empty(device=torch.device("meta"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _first_tensor(output):
    return output if torch.is_tensor(output) else output[0]


def _materialize_rotary(base_model: nn.Module, config, device: torch.device) -> None:
    if not hasattr(base_model, "rotary_emb"):
        return
    rotary_class = type(base_model.rotary_emb)
    try:
        base_model.rotary_emb = rotary_class(config=config).to(device)
    except TypeError:
        base_model.rotary_emb = rotary_class(config).to(device)


def infer_model_family(config, requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(config, "model_type", "") in {"qwen3_vl", "qwen3_vl_moe"}:
        return "qwen3_vl_language"
    return "decoder"


def build_execution_plan(config, dtype: torch.dtype, requested_family: str) -> ExecutionPlan:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("Streamed PPL requires transformers") from error

    family = infer_model_family(config, requested_family)
    if hasattr(config, "use_cache"):
        config.use_cache = False
    with torch.device("meta"):
        if family == "decoder":
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, dtype=dtype
            )
            model.eval()
            if not hasattr(model, "model") or not hasattr(model.model, "layers"):
                raise TypeError("decoder streaming requires model.model.layers")
            return ExecutionPlan(
                family=family,
                forward_model=model,
                base_model=model.model,
                lm_head=model.lm_head,
                config=config,
                hidden_size=config.hidden_size,
                embed_prefix="model.embed_tokens",
                layer_prefix_template="model.layers.{layer}",
                norm_prefix="model.norm",
                lm_head_prefix="lm_head",
                tied_lm_head_prefix="model.embed_tokens",
            )
        if family == "qwen3_vl_language":
            text_config = getattr(config, "text_config", None)
            if text_config is None:
                raise TypeError("Qwen3-VL language streaming requires config.text_config")
            text_config.use_cache = False
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
                Qwen3VLMoeTextModel,
            )

            base_model = Qwen3VLMoeTextModel(text_config)
            base_model.eval()
            lm_head = nn.Linear(
                text_config.hidden_size,
                text_config.vocab_size,
                bias=False,
                dtype=dtype,
            )
            return ExecutionPlan(
                family=family,
                forward_model=base_model,
                base_model=base_model,
                lm_head=lm_head,
                config=text_config,
                hidden_size=text_config.hidden_size,
                embed_prefix="model.language_model.embed_tokens",
                layer_prefix_template="model.language_model.layers.{layer}",
                norm_prefix="model.language_model.norm",
                lm_head_prefix="lm_head",
                tied_lm_head_prefix="model.language_model.embed_tokens",
            )
    raise ValueError(f"Unsupported model family: {family}")


def validate_stream_structure(plan: ExecutionPlan, weight_map: dict[str, str]) -> dict:
    prefixes = {
        "embedding": plan.embed_prefix,
        "first_layer": plan.layer_prefix_template.format(layer=0),
        "last_layer": plan.layer_prefix_template.format(
            layer=len(plan.base_model.layers) - 1
        ),
        "norm": plan.norm_prefix,
    }
    if grouped_names_for_prefix(weight_map, plan.lm_head_prefix):
        prefixes["lm_head"] = plan.lm_head_prefix
    elif grouped_names_for_prefix(weight_map, plan.tied_lm_head_prefix):
        prefixes["lm_head"] = plan.tied_lm_head_prefix
    else:
        raise KeyError("Neither lm_head nor tied embedding weights are present")
    summary = {}
    for label, prefix in prefixes.items():
        grouped = grouped_names_for_prefix(weight_map, prefix)
        if not grouped:
            raise KeyError(f"Checkpoint prefix missing: {prefix}")
        summary[label] = {
            "prefix": prefix,
            "tensor_count": sum(len(names) for names in grouped.values()),
            "shard_count": len(grouped),
        }
    return summary


@torch.no_grad()
def _capture_hidden_states(
    plan: ExecutionPlan,
    checkpoint: Path,
    weight_map: dict[str, str],
    input_ids: torch.Tensor,
    windows: int,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
):
    base_model = plan.base_model
    layers = base_model.layers
    embed_state = load_state_for_prefix(
        checkpoint, weight_map, plan.embed_prefix, dtype
    )
    base_model.embed_tokens = _assign_module(
        base_model.embed_tokens, embed_state, plan.embed_prefix, device
    )
    hidden = torch.empty(
        (windows, sequence_length, plan.hidden_size), dtype=dtype, device="cpu"
    )
    captured: dict = {"index": 0, "kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            for attribute in ("attention_type", "layer_type"):
                if hasattr(original, attribute):
                    setattr(self, attribute, getattr(original, attribute))

        def forward(self, hidden_states, **kwargs):
            index = captured["index"]
            hidden[index].copy_(hidden_states.squeeze(0).detach().cpu())
            captured["index"] = index + 1
            if index == 0:
                captured["kwargs"] = {
                    key: _move(value, "cpu") for key, value in kwargs.items()
                }
            raise _CaptureInterrupt

    original_layer = layers[0]
    layers[0] = Catcher(original_layer)
    try:
        for window in range(windows):
            start = window * sequence_length
            batch = input_ids[:, start : start + sequence_length].to(device)
            try:
                plan.forward_model(batch, use_cache=False)
            except _CaptureInterrupt:
                pass
    finally:
        layers[0] = original_layer
        unload_to_meta(base_model.embed_tokens)
    if captured["index"] != windows:
        raise RuntimeError(
            f"Captured {captured['index']} PPL windows, expected {windows}"
        )
    return hidden, captured["kwargs"]


def _layer_kwargs(layer: nn.Module, captured: dict, device: torch.device) -> dict:
    accepted = set(inspect.signature(layer.forward).parameters)
    return {
        key: _move(value, device)
        for key, value in captured.items()
        if key in accepted
    }


@torch.no_grad()
def streamed_layer_smoke(
    *,
    checkpoint: str | Path,
    input_ids: torch.Tensor,
    sequence_length: int,
    max_layers: int,
    device: str,
    dtype: torch.dtype,
    model_family: str = "auto",
    weight_quantizer,
    activation_config: ActivationQuantizationConfig,
    progress: bool = True,
) -> StreamedLayerSmokeResult:
    """Quantize and execute the first N streamed layers without reporting PPL."""
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError("Streamed layer smoke requires transformers") from error

    if max_layers <= 0:
        raise ValueError("max_layers must be positive")
    checkpoint = Path(checkpoint)
    index = load_safetensors_index(checkpoint)
    weight_map = index["weight_map"]
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    plan = build_execution_plan(config, dtype, model_family)
    validate_stream_structure(plan, weight_map)
    total_layers = len(plan.base_model.layers)
    if total_layers < max_layers:
        raise ValueError(
            f"Requested {max_layers} smoke layers, but the model has {total_layers}"
        )
    if input_ids.shape[1] < sequence_length:
        raise ValueError("No complete input window is available for layer smoke")

    dev = torch.device(device)
    _materialize_rotary(plan.base_model, plan.config, dev)
    hidden, captured = _capture_hidden_states(
        plan,
        checkpoint,
        weight_map,
        input_ids,
        windows=1,
        sequence_length=sequence_length,
        dtype=dtype,
        device=dev,
    )
    outputs = torch.empty_like(hidden)
    wrapped_count = 0
    quantized_count = 0
    layer_results: list[dict] = []

    for layer_index in range(max_layers):
        layer_started = time.perf_counter()
        layer = plan.base_model.layers[layer_index]
        prefix = plan.layer_prefix_template.format(layer=layer_index)
        state = load_state_for_prefix(checkpoint, weight_map, prefix, dtype)
        layer_quantized = quantize_state_weights(state, prefix, weight_quantizer)
        if layer_quantized <= 0:
            raise RuntimeError(f"No paper-scope weights were quantized for {prefix}")
        quantized_count += layer_quantized
        layer = _assign_module(layer, state, prefix, dev)
        layer_wrapped = len(
            wrap_activation_linears(
                layer,
                activation_config,
                prefix=prefix,
                require_lm_head=False,
            )
        )
        if layer_wrapped <= 0:
            raise RuntimeError(f"No activation Linear modules were wrapped for {prefix}")
        wrapped_count += layer_wrapped
        kwargs = _layer_kwargs(layer, captured, dev)
        output = _first_tensor(layer(hidden[0].unsqueeze(0).to(dev), **kwargs))
        output_finite = bool(torch.isfinite(output).all().item())
        if not output_finite:
            raise FloatingPointError(f"Non-finite hidden state at smoke layer {layer_index}")
        outputs[0].copy_(output.squeeze(0).detach().cpu())
        unload_to_meta(layer)
        hidden, outputs = outputs, hidden
        layer_results.append(
            {
                "layer_index": layer_index,
                "prefix": prefix,
                "quantized_weight_count": layer_quantized,
                "wrapped_linear_count": layer_wrapped,
                "output_finite": output_finite,
                "elapsed_seconds": time.perf_counter() - layer_started,
            }
        )
        if progress:
            print(
                f"[streamed-layer-smoke] layer {layer_index + 1}/{max_layers} passed",
                flush=True,
            )

    return StreamedLayerSmokeResult(
        requested_layers=max_layers,
        completed_layers=len(layer_results),
        total_layers=total_layers,
        wrapped_linear_count=wrapped_count,
        quantized_weight_count=quantized_count,
        model_family=plan.family,
        layers=tuple(layer_results),
    )


@torch.no_grad()
def streamed_ppl(
    *,
    checkpoint: str | Path,
    input_ids: torch.Tensor,
    sequence_length: int,
    num_samples: int,
    device: str,
    dtype: torch.dtype,
    model_family: str = "auto",
    weight_quantizer=None,
    activation_config: ActivationQuantizationConfig | None = None,
    progress: bool = True,
) -> StreamedPPLResult:
    """Evaluate PPL by loading one language-model layer at a time."""
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError("Streamed PPL requires transformers") from error

    checkpoint = Path(checkpoint)
    index = load_safetensors_index(checkpoint)
    weight_map = index["weight_map"]
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    plan = build_execution_plan(config, dtype, model_family)
    validate_stream_structure(plan, weight_map)

    available = input_ids.shape[1] // sequence_length
    windows = available if num_samples == 0 else min(num_samples, available)
    if windows <= 0:
        raise ValueError("No complete PPL windows are available")
    dev = torch.device(device)
    _materialize_rotary(plan.base_model, plan.config, dev)
    hidden, captured = _capture_hidden_states(
        plan,
        checkpoint,
        weight_map,
        input_ids,
        windows,
        sequence_length,
        dtype,
        dev,
    )
    outputs = torch.empty_like(hidden)
    wrapped_count = 0
    quantized_count = 0

    for layer_index, layer in enumerate(plan.base_model.layers):
        prefix = plan.layer_prefix_template.format(layer=layer_index)
        state = load_state_for_prefix(checkpoint, weight_map, prefix, dtype)
        if weight_quantizer is not None:
            quantized_count += quantize_state_weights(state, prefix, weight_quantizer)
        layer = _assign_module(layer, state, prefix, dev)
        if activation_config is not None:
            wrapped_count += len(
                wrap_activation_linears(
                    layer,
                    activation_config,
                    prefix=prefix,
                    require_lm_head=False,
                )
            )
        kwargs = _layer_kwargs(layer, captured, dev)
        for window in range(windows):
            output = _first_tensor(layer(hidden[window].unsqueeze(0).to(dev), **kwargs))
            if not torch.isfinite(output).all():
                raise FloatingPointError(
                    f"Non-finite hidden state at layer {layer_index}, window {window}"
                )
            outputs[window].copy_(output.squeeze(0).detach().cpu())
        unload_to_meta(layer)
        hidden, outputs = outputs, hidden
        if progress:
            print(
                f"[streamed-ppl] layer {layer_index + 1}/{len(plan.base_model.layers)}",
                flush=True,
            )

    norm = getattr(plan.base_model, "norm", None)
    if norm is not None:
        norm_state = load_state_for_prefix(checkpoint, weight_map, plan.norm_prefix, dtype)
        plan.base_model.norm = _assign_module(norm, norm_state, plan.norm_prefix, dev)

    if grouped_names_for_prefix(weight_map, plan.lm_head_prefix):
        lm_prefix = plan.lm_head_prefix
    elif getattr(config, "tie_word_embeddings", False):
        lm_prefix = plan.tied_lm_head_prefix
    else:
        raise KeyError("lm_head.weight is absent and tie_word_embeddings is false")
    lm_state = load_state_for_prefix(checkpoint, weight_map, lm_prefix, dtype)
    if weight_quantizer is not None:
        weight = lm_state.get("weight")
        if weight is None:
            raise KeyError(f"{lm_prefix}.weight is absent")
        lm_state["weight"] = weight_quantizer.fake_quantize_weight(
            weight, "lm_head"
        ).contiguous()
        quantized_count += 1
    plan.lm_head = _assign_module(plan.lm_head, lm_state, lm_prefix, dev)
    if activation_config is not None:
        plan.lm_head = ActivationFakeQuantLinear(
            plan.lm_head, name="lm_head", config=activation_config
        )
        wrapped_count += 1

    loss_function = nn.CrossEntropyLoss()
    nll = torch.zeros((), dtype=torch.float32, device=dev)
    for window in range(windows):
        state = hidden[window].unsqueeze(0).to(dev)
        if getattr(plan.base_model, "norm", None) is not None:
            state = plan.base_model.norm(state)
        logits = plan.lm_head(state)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits in PPL window {window}")
        start = window * sequence_length
        labels = input_ids[:, start : start + sequence_length].to(dev)
        loss = loss_function(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
        )
        nll += loss.float() * (sequence_length - 1)

    total_nll = float(nll.item())
    return StreamedPPLResult(
        perplexity=math.exp(total_nll / float(windows * sequence_length)),
        windows=windows,
        total_nll=total_nll,
        wrapped_linear_count=wrapped_count,
        quantized_weight_count=quantized_count,
        model_family=plan.family,
    )
