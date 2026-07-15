#!/usr/bin/env python3
"""Run one paper-aligned downstream benchmark on BF16 or MBPriorQ."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software"))

import torch

from mbpriorq_ae.integration import ActivationQuantizationConfig, wrap_activation_linears
from mbpriorq_ae.logging import set_log_level


GSM8K_PROMPT = (
    "Solve the following math problem step by step. When you have the final answer, "
    'write it as "The answer is $ANSWER$" where $ANSWER$ is the numerical answer.\n\n'
)
MMLU_PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}"""
MMLU_PRO_INITIAL = (
    "The following are multiple choice questions (with answers) about {$}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n\n"
)
CHOICES = tuple("ABCDEFGHIJKLMNOP")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--method", required=True, choices=("bf16", "mbpriorq"))
    parser.add_argument("--model-type", default="cloud", choices=("cloud", "edge"))
    parser.add_argument("--benchmark", required=True, choices=("gsm8k", "mmlu", "mmlu_pro"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-backend", default="full_gpu", choices=("full_gpu", "auto"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_gsm8k(path: Path, limit: int, seed: int):
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError("GSM8K loading requires datasets") from error
    files = [path] if path.is_file() else sorted(path.rglob("test-*.parquet"))
    files = [file for file in files if "socratic" not in file.parts]
    if not files:
        raise FileNotFoundError(f"No GSM8K test parquet found under {path}")
    rows = [dict(row) for row in Dataset.from_parquet(str(files[0]))]
    count = limit or 500
    return random.Random(seed).sample(rows, min(count, len(rows)))


def _load_mmlu(path: Path, limit: int, seed: int):
    csv_path = path if path.is_file() else path / "mmlu.csv"
    if not csv_path.is_file():
        matches = list(path.rglob("mmlu.csv")) if path.is_dir() else []
        if not matches:
            raise FileNotFoundError(f"No mmlu.csv found under {path}")
        csv_path = matches[0]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    count = limit or 100
    return random.Random(seed).sample(rows, min(count, len(rows)))


def _format_mmlu_pro_example(example: dict, include_answer: bool) -> str:
    text = "Question:\n" + example["question"] + "\nOptions:\n"
    for index, option in enumerate(example["options"]):
        text += f"{CHOICES[index]}. {option}\n"
    if include_answer:
        cot = example.get("cot_content", "").replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        return text + cot + "\n\n"
    return text + "Answer: Let's think step by step."


def _load_mmlu_pro(path: Path, limit: int, _seed: int):
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("MMLU-Pro loading requires datasets") from error
    dataset = load_from_disk(str(path))
    category = "computer science"
    validation = [dict(row) for row in dataset["validation"] if row["category"] == category]
    tests = [dict(row) for row in dataset["test"] if row["category"] == category]
    tests = tests[:limit] if limit else tests
    examples = []
    for test in tests:
        prompt = MMLU_PRO_INITIAL.replace("{$}", category) + "\n"
        for example in validation[:5]:
            prompt += _format_mmlu_pro_example(example, True)
        prompt += _format_mmlu_pro_example(test, False)
        item = dict(test)
        item["prompt"] = prompt
        examples.append(item)
    return examples


def _load_examples(args):
    loaders = {
        "gsm8k": _load_gsm8k,
        "mmlu": _load_mmlu,
        "mmlu_pro": _load_mmlu_pro,
    }
    return loaders[args.benchmark](Path(args.dataset).expanduser(), args.num_examples, args.seed)


def _validate_mbpriorq_checkpoint(path: Path) -> dict:
    metadata_path = path / "mbpriorq_ae_prequant_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Downstream MBPriorQ requires an AE-generated checkpoint; missing "
            + str(metadata_path)
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("method") != "mbpriorq" or not metadata.get("lm_head_quantized"):
        raise ValueError("Checkpoint metadata does not describe paper-spec MBPriorQ weights")
    return metadata


def _load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True
    )
    kwargs = {
        "dtype": torch.bfloat16,
        "use_safetensors": True,
        "trust_remote_code": True,
    }
    if args.load_backend == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.load_backend == "full_gpu":
        model = model.to("cuda")
    model.eval()
    wrapped = []
    metadata = None
    if args.method == "mbpriorq":
        metadata = _validate_mbpriorq_checkpoint(Path(args.model))
        wrapped = wrap_activation_linears(
            model,
            ActivationQuantizationConfig(
                method="mbpriorq",
                model_type=args.model_type,
            ),
            preserve_module_forward=True,
        )
    return model, tokenizer, wrapped, metadata


def _input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, args, example_index: int) -> str:
    seed = args.seed + example_index
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_text = f"User: {prompt}\n\nAssistant:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Assistant:")[-1].strip()


def _last_number(text: str):
    matches = re.findall(r"\d+\.?\d*", text.replace(",", ""))
    return float(matches[-1]) if matches else None


def _mmlu_answer(text: str):
    patterns = (
        r"(?i)answer\s*:\s*([A-D])",
        r"(?i)(?:the\s+)?answer\s+is\s+([A-D])",
        r"(?i)([A-D])\.?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def _mmlu_pro_answer(text: str):
    for pattern in (r"(?i)answer is \(?([A-J])\)?", r"(?i)answer:\s*([A-J])"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    matches = re.findall(r"\b([A-J])\b", text, re.IGNORECASE)
    return matches[-1].upper() if matches else None


def _prompt_and_score(benchmark: str, example: dict, response: str, seed: int):
    if benchmark == "gsm8k":
        prediction = _last_number(response)
        gold = _last_number(str(example["answer"]))
        return GSM8K_PROMPT + "\nQuestion: " + example["question"] + "\n", prediction, gold, (
            prediction is not None and gold is not None and abs(prediction - gold) < 1e-6
        )
    if benchmark == "mmlu":
        prompt = MMLU_PROMPT.format(**example)
        prediction = _mmlu_answer(response)
        gold = example["Answer"].strip().upper()
        return prompt, prediction, gold, prediction == gold
    prompt = example["prompt"]
    prediction = _mmlu_pro_answer(response)
    gold = example["answer"].strip().upper()
    if prediction is None:
        # The submitted evaluator used a random choice when extraction failed.
        prediction = CHOICES[random.Random(seed).randrange(len(example["options"]))]
    return prompt, prediction, gold, prediction == gold


def _load_completed(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _prompt_for_example(benchmark: str, example: dict) -> str:
    if benchmark == "gsm8k":
        return GSM8K_PROMPT + "\nQuestion: " + example["question"] + "\n"
    if benchmark == "mmlu":
        return MMLU_PROMPT.format(**example)
    return example["prompt"]


def main():
    args = parse_args()
    set_log_level("release")
    if not torch.cuda.is_available():
        raise RuntimeError("The downstream workflow requires CUDA")
    examples = _load_examples(args)
    model, tokenizer, wrapped, checkpoint_metadata = _load_model(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_suffix(".jsonl")
    records = _load_completed(journal) if args.resume else []
    if len(records) > len(examples):
        raise ValueError("Resume journal has more rows than the selected dataset")
    for index, record in enumerate(records):
        if record.get("index") != index:
            raise ValueError("Resume journal indices are not contiguous from zero")

    started = time.time()

    # Activation priors are runtime state. Replaying completed MBPriorQ prompts
    # restores that state before appending new records; skipping them would make
    # a resumed run numerically different from an uninterrupted run.
    if args.method == "mbpriorq" and records:
        print(f"[resume] replaying {len(records)} prompts to restore activation priors")
        for index, example in enumerate(examples[: len(records)]):
            _generate(model, tokenizer, _prompt_for_example(args.benchmark, example), args, index)

    with journal.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, example in enumerate(examples[len(records) :], start=len(records)):
            prompt = _prompt_for_example(args.benchmark, example)
            response = _generate(model, tokenizer, prompt, args, index)
            _, prediction, gold, correct = _prompt_and_score(
                args.benchmark, example, response, args.seed + index
            )
            record = {
                "index": index,
                "prediction": prediction,
                "gold": gold,
                "correct": bool(correct),
                "response": response,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            records.append(record)
            print(
                f"[{args.benchmark}] {index + 1}/{len(examples)} "
                f"accuracy={sum(row['correct'] for row in records) / len(records):.4f}",
                flush=True,
            )

    correct = sum(record["correct"] for record in records)
    result = {
        "benchmark": args.benchmark,
        "method": args.method,
        "model": args.model,
        "num_examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "load_backend": args.load_backend,
        "wrapped_linear_count": len(wrapped),
        "weight_ebw_summary": (
            checkpoint_metadata.get("weight_ebw_summary") if checkpoint_metadata else None
        ),
        "elapsed_seconds": time.time() - started,
        "records_journal": journal.name,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
