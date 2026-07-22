import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


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


def test_mmlu_pro_answer_does_not_cross_line_boundaries():
    assert RUNNER._mmlu_pro_answer("Answer:\nAnswer: D") == "D"


def test_mmlu_pro_prompt_reduces_demonstrations_to_keep_target():
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": list(range(len(text)))}

    demonstrations = [
        {
            "question": f"demo {index}",
            "options": ["one", "two"],
            "cot_content": "A: Let's think step by step. " + "x" * 80,
        }
        for index in range(5)
    ]
    example = {
        "question": "target question",
        "options": ["one", "two"],
        "category": "computer science",
        "_mmlu_pro_demonstrations": demonstrations,
    }
    prompt, count, token_count = RUNNER._select_mmlu_pro_prompt(
        example, CharacterTokenizer(), max_input_tokens=500
    )

    assert count < 5
    assert token_count <= 500
    assert "target question" in prompt


def test_generate_decodes_only_new_tokens():
    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, tokens, skip_special_tokens=True):
            return ",".join(str(int(token)) for token in tokens)

    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 9, 10]])

    args = SimpleNamespace(
        seed=0,
        benchmark="mmlu",
        max_new_tokens=2,
        temperature=0.1,
        top_p=0.9,
    )
    assert RUNNER._generate(FakeModel(), FakeTokenizer(), "prompt", args, 0) == "9,10"


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
