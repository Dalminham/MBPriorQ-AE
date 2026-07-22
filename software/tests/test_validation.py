import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "software/tools"


def _run(tool, *args):
    return subprocess.run(
        [sys.executable, str(TOOLS / tool), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_quick_ppl_rejects_nonfinite_value(tmp_path):
    expected = tmp_path / "expected.csv"
    expected.write_text("row,perplexity,tolerance,num_samples\nbf16,1,1,1\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "bf16.json").write_text(
        json.dumps({"method": "bf16", "num_samples": 1, "perplexity": float("nan"), "total_nll": 1}),
        encoding="utf-8",
    )
    completed = _run(
        "validate_ppl_results.py",
        "--results", results,
        "--expected", expected,
        "--expected-samples", "1",
    )
    assert completed.returncode != 0
    assert "finite" in completed.stderr


def test_quick_ppl_rejects_wrong_sample_count(tmp_path):
    expected = tmp_path / "expected.csv"
    expected.write_text("row,perplexity,tolerance,num_samples\nbf16,1,1,1\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "bf16.json").write_text(
        json.dumps({"method": "bf16", "num_samples": 2, "perplexity": 1.0, "total_nll": 1.0}),
        encoding="utf-8",
    )
    completed = _run(
        "validate_ppl_results.py",
        "--results", results,
        "--expected", expected,
        "--expected-samples", "1",
    )
    assert completed.returncode != 0
    assert "expected 1" in completed.stderr


def test_downstream_quick_validates_journal(tmp_path):
    expected = tmp_path / "expected.csv"
    expected.write_text(
        "model_key,model,method,gsm8k\nmodel,Model,bf16,0\nmodel,Model,mbpriorq,0\n",
        encoding="utf-8",
    )
    for method in ("bf16", "mbpriorq"):
        journal = tmp_path / f"model__{method}__gsm8k.jsonl"
        journal.write_text(
            json.dumps({"index": 0, "prediction": 1, "gold": 1, "correct": True, "response": "1"}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / f"model__{method}__gsm8k.json").write_text(
            json.dumps(
                {
                    "benchmark": "gsm8k",
                    "method": method,
                    "num_examples": 1,
                    "correct": 1,
                    "accuracy": 1.0,
                    "records_journal": journal.name,
                }
            ),
            encoding="utf-8",
        )
    completed = _run(
        "validate_downstream_results.py",
        "--output-root", tmp_path,
        "--expected", expected,
        "--model-keys", "model",
        "--benchmarks", "gsm8k",
        "--expected-examples", "1",
    )
    assert completed.returncode == 0, completed.stderr
    assert "passed for 2 rows" in completed.stdout


def test_downstream_quick_rejects_missing_journal_row(tmp_path):
    expected = tmp_path / "expected.csv"
    expected.write_text(
        "model_key,model,method,gsm8k\nmodel,Model,bf16,0\n",
        encoding="utf-8",
    )
    journal = tmp_path / "model__bf16__gsm8k.jsonl"
    journal.write_text("", encoding="utf-8")
    (tmp_path / "model__bf16__gsm8k.json").write_text(
        json.dumps(
            {
                "benchmark": "gsm8k",
                "method": "bf16",
                "num_examples": 1,
                "correct": 0,
                "accuracy": 0.0,
                "records_journal": journal.name,
            }
        ),
        encoding="utf-8",
    )
    completed = _run(
        "validate_downstream_results.py",
        "--output-root", tmp_path,
        "--expected", expected,
        "--model-keys", "model",
        "--benchmarks", "gsm8k",
        "--expected-examples", "1",
    )
    assert completed.returncode != 0
    assert "journal rows" in completed.stderr


def _write_full_mmlu_result(tmp_path, correct):
    journal = tmp_path / "model__bf16__mmlu.jsonl"
    journal.write_text(
        "".join(
            json.dumps(
                {
                    "index": index,
                    "prediction": "A",
                    "gold": "A" if index < correct else "B",
                    "correct": index < correct,
                    "response": "Answer: A",
                }
            )
            + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    (tmp_path / "model__bf16__mmlu.json").write_text(
        json.dumps(
            {
                "benchmark": "mmlu",
                "method": "bf16",
                "num_examples": 100,
                "correct": correct,
                "accuracy": correct / 100,
                "records_journal": journal.name,
            }
        ),
        encoding="utf-8",
    )
    expected = tmp_path / "expected.csv"
    expected.write_text(
        "model_key,model,method,mmlu\nmodel,Model,bf16,41\n",
        encoding="utf-8",
    )
    return expected


def test_full_mmlu_accepts_two_percentage_point_difference(tmp_path):
    expected = _write_full_mmlu_result(tmp_path, correct=39)
    completed = _run(
        "validate_downstream_results.py",
        "--output-root", tmp_path,
        "--expected", expected,
        "--model-keys", "model",
        "--benchmarks", "mmlu",
        "--require-full",
    )
    assert completed.returncode == 0, completed.stderr
    assert "tolerance=2.00 pp" in completed.stdout


def test_full_mmlu_rejects_more_than_two_percentage_points(tmp_path):
    expected = _write_full_mmlu_result(tmp_path, correct=38)
    completed = _run(
        "validate_downstream_results.py",
        "--output-root", tmp_path,
        "--expected", expected,
        "--model-keys", "model",
        "--benchmarks", "mmlu",
        "--require-full",
    )
    assert completed.returncode != 0
    assert "3.00 pp exceeds 2.00 pp" in completed.stderr
