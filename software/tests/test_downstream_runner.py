import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "software/tools/run_downstream_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_downstream_benchmark", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _mmlu_example(answer="D"):
    return {
        "Question": "Example?",
        "A": "one",
        "B": "two",
        "C": "three",
        "D": "four",
        "Answer": answer,
    }


def test_mmlu_answer_does_not_treat_next_answer_word_as_option_a():
    assert RUNNER._mmlu_answer("Answer:\nAnswer: D") == "D"
    assert RUNNER._mmlu_answer("Answer:\nExplanation without a final choice") is None


def test_mmlu_retry_runs_once_with_the_retry_seed(monkeypatch):
    calls = []

    def fake_generate(model, tokenizer, prompt, args, example_index, attempt=0):
        calls.append((example_index, attempt))
        return "Answer: D"

    monkeypatch.setattr(RUNNER, "_generate", fake_generate)
    record = {
        "index": 7,
        "prediction": None,
        "gold": "D",
        "correct": False,
        "response": "No final selection.",
    }
    changed = RUNNER._retry_mmlu_record(
        object(), object(), SimpleNamespace(seed=0), _mmlu_example(), record
    )

    assert changed is True
    assert calls == [(7, 1)]
    assert record["generation_attempts"] == 2
    assert record["prediction"] == "D"
    assert record["correct"] is True

    assert (
        RUNNER._retry_mmlu_record(
            object(), object(), SimpleNamespace(seed=0), _mmlu_example(), record
        )
        is False
    )
    assert calls == [(7, 1)]
