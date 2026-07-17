"""Paper-compatible WikiText2 perplexity evaluation."""

from __future__ import annotations

import gc
import inspect
import math
from pathlib import Path

import torch
import torch.nn as nn


class _CaptureInterrupt(Exception):
    pass


def load_text_dataset(dataset: str, split: str = "test"):
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as error:
        raise RuntimeError("PPL evaluation requires the datasets package") from error

    path = Path(dataset).expanduser()
    if path.exists():
        loaded = load_from_disk(str(path))
    elif dataset in {"wikitext-2", "wikitext-2-raw-v1"}:
        loaded = load_dataset("wikitext", "wikitext-2-raw-v1")
    else:
        loaded = load_dataset(dataset)
    return loaded[split] if hasattr(loaded, "keys") else loaded


def encode_dataset(tokenizer, dataset) -> torch.Tensor:
    for column in ("text", "sentence", "content"):
        if column in dataset.column_names:
            text = "\n\n".join(dataset[column])
            return tokenizer(text, return_tensors="pt", verbose=False).input_ids
    raise ValueError(f"No text column in dataset columns {dataset.column_names}")


def _embedding_layer(model):
    for name in ("embed_tokens", "word_embeddings", "embeddings", "token_embeddings"):
        if hasattr(model.model, name):
            return name, getattr(model.model, name)
    raise AttributeError("Could not locate the model embedding layer")


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported transformer-layer output: {type(output).__name__}")


@torch.no_grad()
def paper_compatible_full_model_ppl(
    model,
    input_ids: torch.Tensor,
    *,
    sequence_length: int = 2048,
    num_samples: int = 0,
    batch_size: int = 1,
    device: str = "cuda",
    progress: bool = True,
    cache_factory=None,
) -> tuple[float, int, float]:
    """Evaluate contiguous WikiText2 windows with the whole model resident."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected token ids with shape [1, tokens], got {tuple(input_ids.shape)}")
    available = input_ids.shape[1] // sequence_length
    windows = available if num_samples == 0 else min(int(num_samples), available)
    if windows <= 0:
        raise ValueError(
            f"Dataset has {input_ids.shape[1]} tokens, fewer than one {sequence_length}-token window"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    original_use_cache = getattr(model.config, "use_cache", None)
    use_cache = cache_factory is not None
    if original_use_cache is not None:
        model.config.use_cache = use_cache
    loss_function = nn.CrossEntropyLoss()
    nll = torch.zeros((), dtype=torch.float32, device=dev)
    completed_windows = 0
    for first_window in range(0, windows, batch_size):
        current_batch_size = min(batch_size, windows - first_window)
        batches = []
        for offset in range(current_batch_size):
            start = (first_window + offset) * sequence_length
            batches.append(input_ids[:, start : start + sequence_length])
        batch = torch.cat(batches, dim=0).to(dev)
        model_kwargs = {"use_cache": use_cache}
        if cache_factory is not None:
            model_kwargs["past_key_values"] = cache_factory()
        logits = model(batch, **model_kwargs).logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"Non-finite logits in PPL batch starting at window {first_window}"
            )
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = batch[:, 1:]
        loss = loss_function(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
        )
        nll += loss.float() * (current_batch_size * (sequence_length - 1))
        completed_windows += current_batch_size
        if progress:
            print(f"[ppl] window {completed_windows}/{windows}", flush=True)
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache
    total_nll = float(nll.item())
    return math.exp(total_nll / float(windows * sequence_length)), windows, total_nll


@torch.no_grad()
def paper_compatible_kv_cache_ppl(
    model,
    input_ids: torch.Tensor,
    *,
    sequence_length: int = 2048,
    num_samples: int = 0,
    device: str = "cuda",
    progress: bool = True,
    cache_factory=None,
) -> tuple[float, int, float]:
    """Evaluate the full text stream with the Table 8 KV-cache protocol."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected token ids with shape [1, tokens], got {tuple(input_ids.shape)}")

    starts = list(range(0, input_ids.shape[1], sequence_length))
    if num_samples > 0:
        starts = starts[: int(num_samples)]
    if not starts:
        raise ValueError("The dataset contains no tokens")

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    nll = torch.zeros((), dtype=torch.float32, device=dev)
    loss_tokens = 0
    chunks = 0
    for start in starts:
        batch = input_ids[:, start : start + sequence_length].to(dev)
        if batch.shape[1] < 2:
            continue
        model_kwargs = {}
        if cache_factory is not None:
            model_kwargs["past_key_values"] = cache_factory()
            model_kwargs["use_cache"] = True
        logits = model(batch, **model_kwargs).logits
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits in KV-cache chunk {chunks}")
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = batch[:, 1:]
        current_tokens = int(shifted_labels.numel())
        loss = loss_function(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
        )
        nll += loss.float() * current_tokens
        loss_tokens += current_tokens
        chunks += 1
        if progress:
            print(f"[ppl] KV-cache chunk {chunks}/{len(starts)}", flush=True)
    if loss_tokens == 0:
        raise ValueError("The selected KV-cache PPL chunks contain no next-token targets")
    total_nll = float(nll.item())
    return math.exp(total_nll / float(loss_tokens)), chunks, total_nll


@torch.no_grad()
def paper_compatible_by_layer_ppl(
    model,
    input_ids: torch.Tensor,
    *,
    sequence_length: int = 2048,
    num_samples: int = 0,
    device: str = "cuda",
    progress: bool = True,
) -> tuple[float, int, float]:
    """Reproduce the paper's contiguous-window, layer-wise PPL calculation.

    The denominator intentionally remains ``windows * sequence_length`` to
    match the submitted EasyLLM evaluation, whose cross-entropy numerator uses
    ``sequence_length - 1`` next-token targets per window.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected token ids with shape [1, tokens], got {tuple(input_ids.shape)}")
    available = input_ids.shape[1] // sequence_length
    windows = available if num_samples == 0 else min(int(num_samples), available)
    if windows <= 0:
        raise ValueError(
            f"Dataset has {input_ids.shape[1]} tokens, fewer than one {sequence_length}-token window"
        )
    if not hasattr(model.model, "layers"):
        raise TypeError("The AE by-layer runner expects model.model.layers")

    dev = torch.device(device)
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    embedding_name, embedding = _embedding_layer(model)
    setattr(model.model, embedding_name, embedding.to(dev))
    layers[0] = layers[0].to(dev)
    dtype = next(model.parameters()).dtype

    inputs = torch.zeros(
        (windows, sequence_length, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    outputs = torch.zeros_like(inputs)
    capture: dict = {"index": 0, "parameters": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.parameter_names = list(inspect.signature(module.forward).parameters)
            for attribute in ("attention_type", "layer_type"):
                if hasattr(module, attribute):
                    setattr(self, attribute, getattr(module, attribute))

        def forward(self, hidden_states, **kwargs):
            index = capture["index"]
            if hidden_states.shape[0] != 1:
                raise ValueError(
                    "The paper-compatible catcher expects one PPL window per forward"
                )
            inputs[index].copy_(hidden_states.squeeze(0))
            capture["index"] = index + 1
            for key, value in kwargs.items():
                if key != "hidden_states":
                    capture["parameters"][key] = value
            raise _CaptureInterrupt

    layers[0] = Catcher(layers[0])
    for window in range(windows):
        start = window * sequence_length
        batch = input_ids[:, start : start + sequence_length].to(dev)
        try:
            model(batch)
        except _CaptureInterrupt:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    setattr(model.model, embedding_name, getattr(model.model, embedding_name).cpu())
    torch.cuda.empty_cache()

    captured_parameters = capture["parameters"]
    for layer_index in range(len(layers)):
        layer = layers[layer_index].to(dev)
        accepted = set(inspect.signature(layer.forward).parameters)
        kwargs = {
            key: value
            for key, value in captured_parameters.items()
            if key in accepted
        }
        for window in range(windows):
            result = _first_tensor(layer(inputs[window].unsqueeze(0), **kwargs))
            if not torch.isfinite(result).all():
                raise FloatingPointError(
                    f"Non-finite hidden state at layer {layer_index}, window {window}"
                )
            if result.shape[0] != 1:
                raise ValueError(
                    f"Expected one output window, got shape {tuple(result.shape)}"
                )
            outputs[window].copy_(result.squeeze(0))
        layer.to_empty(device=torch.device("cpu"))
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        inputs, outputs = outputs, inputs
        if progress:
            print(f"[ppl] layer {layer_index + 1}/{len(layers)}", flush=True)

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    labels = input_ids.to(dev)
    loss_function = nn.CrossEntropyLoss()
    nll = torch.zeros((), dtype=torch.float32, device=dev)
    for window in range(windows):
        hidden = inputs[window].unsqueeze(0)
        if model.model.norm is not None:
            hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
        shifted_logits = logits[:, :-1, :].contiguous()
        start = window * sequence_length
        shifted_labels = labels[:, start : start + sequence_length][:, 1:]
        loss = loss_function(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
        )
        nll += loss.float() * (sequence_length - 1)

    model.config.use_cache = original_use_cache
    total_nll = float(nll.item())
    ppl = math.exp(total_nll / float(windows * sequence_length))
    return ppl, windows, total_nll
