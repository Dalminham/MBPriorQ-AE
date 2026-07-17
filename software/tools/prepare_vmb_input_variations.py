#!/usr/bin/env python3
"""Prepare prompt and input-segment datasets used by the Table 7 workflow."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_split(path: str, split: str):
    dataset = load_from_disk(path)
    return dataset[split] if hasattr(dataset, "keys") else dataset


def _render_prompt(example: dict) -> str:
    options = list(example["options"])
    if len(options) > len(LABELS):
        raise ValueError(f"MMLU-Pro example has {len(options)} options")
    lines = [
        f"Subject: {example['category']}",
        "Question:",
        str(example["question"]),
        "Choices:",
    ]
    lines.extend(f"{LABELS[index]}. {option}" for index, option in enumerate(options))
    lines.append("Answer:")
    return "\n".join(lines)


def _suffix_character_counts(texts: list[str]) -> list[int]:
    counts = [0] * (len(texts) + 1)
    for index in range(len(texts) - 1, -1, -1):
        counts[index] = counts[index + 1] + len(texts[index])
    return counts


def _choose_start(texts: list[str], seed: int, minimum_characters: int) -> tuple[int, int]:
    suffix_counts = _suffix_character_counts(texts)
    candidates = [
        index
        for index, text in enumerate(texts)
        if text.strip() and suffix_counts[index] >= minimum_characters
    ]
    if not candidates:
        raise ValueError(
            f"No WikiText2 start row leaves {minimum_characters} characters"
        )
    start = random.Random(seed).choice(candidates)
    return start, suffix_counts[start]


def _replace_directory(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite")
    shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--mmlu-pro", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wikitext-split", default="test")
    parser.add_argument("--mmlu-pro-split", default="test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--minimum-suffix-characters", type=int, default=262144)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    _replace_directory(output_root, args.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)

    mmlu = _load_split(args.mmlu_pro, args.mmlu_pro_split)
    prompt_path = output_root / "mmlu_pro_prompts"
    prompts = [_render_prompt(example) for example in mmlu]
    DatasetDict({"test": Dataset.from_dict({"text": prompts})}).save_to_disk(prompt_path)

    wikitext = _load_split(args.wikitext, args.wikitext_split)
    texts = [str(text) for text in wikitext["text"]]
    streams = []
    for seed in args.seeds:
        start, remaining_characters = _choose_start(
            texts, seed, args.minimum_suffix_characters
        )
        stream_path = output_root / f"wikitext2_segment_seed_{seed}"
        DatasetDict(
            {args.wikitext_split: Dataset.from_dict({"text": texts[start:]})}
        ).save_to_disk(stream_path)
        streams.append(
            {
                "seed": seed,
                "dataset_path": str(stream_path),
                "start_row": start,
                "remaining_characters": remaining_characters,
            }
        )

    manifest = {
        "wikitext_source": args.wikitext,
        "wikitext_split": args.wikitext_split,
        "mmlu_pro_source": args.mmlu_pro,
        "mmlu_pro_split": args.mmlu_pro_split,
        "prompt_dataset_path": str(prompt_path),
        "prompt_count": len(prompts),
        "prompt_template": "Subject, question, choices, and Answer: without the gold answer",
        "segment_definition": (
            "Each seed selects a deterministic WikiText2 start row and preserves "
            "the original row order from that point onward."
        ),
        "streams": streams,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared Table 7 input datasets: {manifest_path}")


if __name__ == "__main__":
    main()
