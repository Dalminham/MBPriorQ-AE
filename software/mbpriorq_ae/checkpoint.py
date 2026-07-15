"""Streaming fake-quantized checkpoint generation for the AE workflows."""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from .ebw import GlobalEBW
from .integration import make_quantizer


INDEX_NAME = "model.safetensors.index.json"
INCOMPLETE_MARKER = ".mbpriorq_ae_incomplete"
EMBED_TENSOR_NAME = "model.embed_tokens.weight"
LM_HEAD_TENSOR_NAME = "lm_head.weight"


def _safetensors_api():
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise RuntimeError("Checkpoint generation requires the safetensors package") from error
    return safe_open, load_file, save_file


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_index(model_path: Path) -> tuple[dict, bool]:
    safe_open, _, _ = _safetensors_api()
    index_path = model_path / INDEX_NAME
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        if "weight_map" not in index:
            raise ValueError(f"Missing weight_map in {index_path}")
        return index, True

    single_shard = model_path / "model.safetensors"
    if not single_shard.exists():
        raise FileNotFoundError(
            f"Expected {index_path.name} or {single_shard.name} under {model_path}"
        )
    with safe_open(str(single_shard), framework="pt", device="cpu") as handle:
        weight_map = {name: single_shard.name for name in handle.keys()}
    return {
        "metadata": {"total_size": single_shard.stat().st_size},
        "weight_map": weight_map,
    }, False


def _load_config(model_path: Path) -> dict:
    path = model_path / "config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_tensors(weight_map: dict[str, str]) -> tuple[list[str], dict[str, list[str]]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    ordered: list[str] = []
    for tensor_name, shard_name in weight_map.items():
        grouped[shard_name].append(tensor_name)
        if shard_name not in ordered:
            ordered.append(shard_name)
    return ordered, grouped


def _should_quantize(tensor_name: str, tensor: torch.Tensor) -> bool:
    if tensor_name == LM_HEAD_TENSOR_NAME:
        return tensor.ndim >= 2
    if tensor.ndim < 2:
        return False
    regular_weight = tensor_name.endswith(".weight")
    stacked_expert = ".mlp.experts." in tensor_name
    if not (regular_weight or stacked_expert):
        return False
    return any(
        marker in tensor_name
        for marker in (".self_attn.", ".mlp.", ".block_sparse_moe.")
    )


def _module_name(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else tensor_name


def _copy_side_files(source: Path, output: Path) -> None:
    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    for root, _, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        output_root = output / relative
        output_root.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            if file_name == INDEX_NAME or file_name.endswith(skip_suffixes):
                continue
            shutil.copy2(root_path / file_name, output_root / file_name)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _quantize_tensor(quantizer, tensor: torch.Tensor, name: str) -> torch.Tensor:
    with torch.no_grad():
        result = quantizer.fake_quantize_weight(tensor, name)
    if result.shape != tensor.shape or result.dtype != tensor.dtype:
        raise RuntimeError(
            f"Quantizer changed tensor contract for {name}: "
            f"{tuple(tensor.shape)}/{tensor.dtype} -> {tuple(result.shape)}/{result.dtype}"
        )
    return result.contiguous()


def stream_fake_quantize_checkpoint(
    *,
    source_path: str | os.PathLike,
    output_path: str | os.PathLike,
    method: str,
    refined_block_size: int = 4,
    using_imatrix: bool = False,
    imatrix_file_name: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Generate a loadable HF checkpoint while bounding live source data to one shard."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Model directory not found: {source}")
    if method != "mbpriorq":
        raise ValueError(f"Unsupported checkpoint method: {method}")
    if using_imatrix and (not imatrix_file_name or not Path(imatrix_file_name).is_file()):
        raise FileNotFoundError(f"Importance matrix not found: {imatrix_file_name}")

    _, load_file, save_file = _safetensors_api()
    index, had_index = _load_index(source)
    _prepare_output(output, overwrite)
    _copy_side_files(source, output)

    GlobalEBW.reset()
    quantizer = make_quantizer(
        method=method,
        name=None,
        model_type="cloud",
        refined_block_size=refined_block_size,
        using_imatrix=using_imatrix,
        imatrix_file_name=imatrix_file_name,
    )

    weight_map = index["weight_map"]
    shards, grouped = _group_tensors(weight_map)
    config = _load_config(source)
    tied = bool(config.get("tie_word_embeddings", False)) and EMBED_TENSOR_NAME in weight_map
    output_index = json.loads(json.dumps(index))
    if tied:
        output_index["weight_map"].pop(LM_HEAD_TENSOR_NAME, None)

    stats = {
        "method": method,
        "refined_block_size": int(refined_block_size),
        "using_imatrix": bool(using_imatrix),
        "num_shards": len(shards),
        "num_tensors": len(weight_map),
        "num_quantized_tensors": 0,
        "num_quantized_elements": 0,
        "lm_head_quantized": False,
        "tie_word_embeddings": tied,
        "lm_head_saved_as": EMBED_TENSOR_NAME if tied else LM_HEAD_TENSOR_NAME,
    }
    marker = output / INCOMPLETE_MARKER
    marker.write_text("streaming fake quantization in progress\n", encoding="utf-8")
    tied_lm_head: torch.Tensor | None = None

    try:
        for shard_index, shard_name in enumerate(shards, start=1):
            source_shard = source / shard_name
            tensors = load_file(str(source_shard), device="cpu")
            print(f"[checkpoint] shard {shard_index}/{len(shards)}: {shard_name}", flush=True)

            for tensor_name in grouped[shard_name]:
                tensor = tensors[tensor_name]
                if tied and tensor_name == LM_HEAD_TENSOR_NAME:
                    if tied_lm_head is None:
                        print("[checkpoint] quantizing tied lm_head.weight", flush=True)
                        tied_lm_head = _quantize_tensor(quantizer, tensor, "lm_head").clone()
                        stats["lm_head_quantized"] = True
                    del tensors[tensor_name]
                    _release_memory()
                    continue

                if tied and tensor_name == EMBED_TENSOR_NAME:
                    if tied_lm_head is None:
                        print("[checkpoint] quantizing tied lm_head.weight", flush=True)
                        tied_lm_head = _quantize_tensor(quantizer, tensor, "lm_head").clone()
                        stats["lm_head_quantized"] = True
                    tensors[tensor_name] = tied_lm_head.clone()
                    stats["num_quantized_tensors"] += 1
                    stats["num_quantized_elements"] += int(tensor.numel())
                    _release_memory()
                    continue

                if not _should_quantize(tensor_name, tensor):
                    continue

                name = _module_name(tensor_name)
                print(f"[checkpoint] quantizing {name}", flush=True)
                tensors[tensor_name] = _quantize_tensor(quantizer, tensor, name)
                stats["num_quantized_tensors"] += 1
                stats["num_quantized_elements"] += int(tensor.numel())
                if tensor_name == LM_HEAD_TENSOR_NAME:
                    stats["lm_head_quantized"] = True
                _release_memory()

            output_shard = output / shard_name
            output_shard.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(output_shard), metadata={"format": "pt"})
            del tensors
            _release_memory()

        if had_index or len(shards) > 1 or tied:
            with (output / INDEX_NAME).open("w", encoding="utf-8") as handle:
                json.dump(output_index, handle, indent=2, sort_keys=True)
        if not stats["lm_head_quantized"]:
            raise RuntimeError("Paper-spec checkpoint generation did not quantize lm_head")

        stats["weight_ebw_summary"] = GlobalEBW.summarize("weight")
        with (output / "mbpriorq_ae_prequant_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
        marker.unlink()
        return stats
    except Exception:
        marker.write_text("streaming fake quantization failed\n", encoding="utf-8")
        raise
