"""Paper Table 8 KV-cache fake quantization paths."""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer

from .integration import make_quantizer
from .mbpriorq import FP4_MAX, FP8_MAX, SCALING_VECTOR_SIZE


def _pad_last_dim(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    original_length = tensor.shape[-1]
    padding = (-original_length) % multiple
    if padding == 0:
        return tensor, original_length
    zeros = torch.zeros(
        (*tensor.shape[:-1], padding), dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat((tensor, zeros), dim=-1), original_length


def _regular_nvfp4(
    tensor: torch.Tensor,
    quantizer,
) -> torch.Tensor:
    shape = tensor.shape
    flat = tensor.reshape(-1, shape[-1]).contiguous()
    if flat.shape[-1] % SCALING_VECTOR_SIZE != 0:
        raise ValueError(
            f"NVFP4 quantization dimension {flat.shape[-1]} is not divisible by "
            f"{SCALING_VECTOR_SIZE}"
        )
    global_scale = flat.abs().amax().float() / (FP4_MAX * FP8_MAX)
    packed, block_scale = quantizer._quantize_nvfp4(
        flat, SCALING_VECTOR_SIZE, global_scale
    )
    dequantized = quantizer._dequantize_nvfp4(
        packed, block_scale, global_scale, tuple(flat.shape), tensor.dtype
    )
    return dequantized.reshape(shape)


class _KVQuantizedLayer(DynamicLayer):
    def _quantize_key(self, tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        transposed = tensor.transpose(-2, -1).contiguous()
        padded, original_length = _pad_last_dim(transposed, SCALING_VECTOR_SIZE)
        quantized = self._quantize(padded, f"layer_{self.layer_index}.key")
        quantized = quantized[..., :original_length]
        return quantized.transpose(-2, -1).reshape(original_shape)

    def _quantize_value(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] % SCALING_VECTOR_SIZE != 0:
            raise ValueError(
                f"KV-cache head dimension {tensor.shape[-1]} is not divisible by "
                f"{SCALING_VECTOR_SIZE}"
            )
        return self._quantize(tensor, f"layer_{self.layer_index}.value")

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        key_quantized = self._quantize_key(key_states)
        value_quantized = self._quantize_value(value_states)
        self.keys = torch.cat((self.keys, key_quantized), dim=-2)
        self.values = torch.cat((self.values, value_quantized), dim=-2)
        return self.keys, self.values


class NVFP4KVLayer(_KVQuantizedLayer):
    """NVFP4 with K quantized by sequence and V by head dimension."""

    def __init__(self, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.quantizer = make_quantizer(
            method="mbpriorq",
            name=f"kv_cache.layer_{layer_index}.nvfp4",
            model_type="cloud",
            metadata_target="kv_cache",
        )

    def _quantize(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return _regular_nvfp4(tensor, self.quantizer)


class MBPriorQKVLayer(_KVQuantizedLayer):
    """MBPriorQ with K quantized by sequence and V by head dimension."""

    def __init__(self, layer_index: int, refined_block_size: int = 4):
        super().__init__()
        self.layer_index = layer_index
        common = {
            "method": "mbpriorq",
            "model_type": "edge",
            "ablation_mode": "paper",
            "refined_block_size": refined_block_size,
            "metadata_target": "kv_cache",
        }
        self.key_quantizer = make_quantizer(
            name=f"kv_cache.layer_{layer_index}.key", **common
        )
        self.value_quantizer = make_quantizer(
            name=f"kv_cache.layer_{layer_index}.value", **common
        )

    def _quantize(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        quantizer = self.key_quantizer if name.endswith(".key") else self.value_quantizer
        shape = tensor.shape
        flat = tensor.reshape(-1, shape[-1]).contiguous()
        quantized = quantizer.fake_quantize_activation(flat, name, flat.shape)
        return quantized.reshape(shape)


def build_kv_cache(config, method: str, refined_block_size: int = 4) -> Cache:
    """Create a fresh cache for one full-stream PPL chunk."""
    if method not in {"nvfp4", "mbpriorq"}:
        raise ValueError(f"Unsupported KV-cache method: {method}")
    decoder_config = (
        config.get_text_config(decoder=True)
        if hasattr(config, "get_text_config")
        else config
    )
    layer_count = int(decoder_config.num_hidden_layers)
    if method == "nvfp4":
        layers = [NVFP4KVLayer(index) for index in range(layer_count)]
    else:
        layers = [
            MBPriorQKVLayer(index, refined_block_size=refined_block_size)
            for index in range(layer_count)
        ]
    return Cache(layers=layers)
