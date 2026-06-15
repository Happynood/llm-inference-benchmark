"""Tests for task_quality module: rubric evaluation and runner integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_inference_benchmark.task_quality import (
    TaskQualityReport,
    TaskRubric,
    compute_task_quality,
    evaluate_text,
    load_task_rubrics,
)

# ---------------------------------------------------------------------------
# evaluate_text
# ---------------------------------------------------------------------------


def test_evaluate_text_empty_rubric_always_passes() -> None:
    assert evaluate_text("anything", TaskRubric()) is True
    assert evaluate_text("", TaskRubric()) is True


def test_evaluate_text_contains_all_pass() -> None:
    rubric = TaskRubric(contains_all=["hello", "world"])
    assert evaluate_text("Hello World", rubric) is True


def test_evaluate_text_contains_all_fail() -> None:
    rubric = TaskRubric(contains_all=["hello", "missing"])
    assert evaluate_text("Hello World", rubric) is False


def test_evaluate_text_contains_any_pass() -> None:
    rubric = TaskRubric(contains_any=["alpha", "beta"])
    assert evaluate_text("beta release", rubric) is True


def test_evaluate_text_contains_any_fail() -> None:
    rubric = TaskRubric(contains_any=["alpha", "beta"])
    assert evaluate_text("gamma release", rubric) is False


def test_evaluate_text_forbidden_fail() -> None:
    rubric = TaskRubric(forbidden=["I don't know"])
    assert evaluate_text("I don't know the answer.", rubric) is False


def test_evaluate_text_forbidden_pass() -> None:
    rubric = TaskRubric(forbidden=["error"])
    assert evaluate_text("Result is 42", rubric) is True


def test_evaluate_text_forbidden_case_insensitive() -> None:
    rubric = TaskRubric(forbidden=["ERROR"])
    assert evaluate_text("An error occurred", rubric) is False


def test_evaluate_text_regex_pass() -> None:
    rubric = TaskRubric(regex=r"\d+")
    assert evaluate_text("The answer is 42", rubric) is True


def test_evaluate_text_regex_fail() -> None:
    rubric = TaskRubric(regex=r"^\d+$")
    assert evaluate_text("not digits", rubric) is False


def test_evaluate_text_min_chars_pass() -> None:
    rubric = TaskRubric(min_chars=5)
    assert evaluate_text("hello", rubric) is True


def test_evaluate_text_min_chars_fail() -> None:
    rubric = TaskRubric(min_chars=10)
    assert evaluate_text("hi", rubric) is False


def test_evaluate_text_min_chars_strips_whitespace() -> None:
    rubric = TaskRubric(min_chars=5)
    assert evaluate_text("  hi  ", rubric) is False  # stripped len == 2


def test_evaluate_text_case_insensitive_contains_all() -> None:
    rubric = TaskRubric(contains_all=["PARIS"])
    assert evaluate_text("paris is the capital", rubric) is True


# ---------------------------------------------------------------------------
# load_task_rubrics
# ---------------------------------------------------------------------------


def test_load_task_rubrics_valid(tmp_path: Path) -> None:
    spec = tmp_path / "quality.yaml"
    spec.write_text("- contains_any:\n    - mock\n  min_chars: 3\n- null\n- regex: '\\d+'\n")
    rubrics = load_task_rubrics(spec)
    assert len(rubrics) == 3
    assert rubrics[0] is not None
    assert rubrics[0].min_chars == 3
    assert rubrics[1] is None
    assert rubrics[2] is not None
    assert rubrics[2].regex == "\\d+"


def test_load_task_rubrics_empty_list(tmp_path: Path) -> None:
    spec = tmp_path / "empty.yaml"
    spec.write_text("[]\n")
    assert load_task_rubrics(spec) == []


def test_load_task_rubrics_not_a_list(tmp_path: Path) -> None:
    spec = tmp_path / "bad.yaml"
    spec.write_text("key: value\n")
    with pytest.raises(ValueError, match="must be a YAML list"):
        load_task_rubrics(spec)


def test_load_task_rubrics_unknown_key(tmp_path: Path) -> None:
    spec = tmp_path / "bad.yaml"
    spec.write_text("- unknown_key: oops\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_task_rubrics(spec)


def test_load_task_rubrics_not_a_mapping(tmp_path: Path) -> None:
    spec = tmp_path / "bad.yaml"
    spec.write_text("- just a string\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_task_rubrics(spec)


def test_load_task_rubrics_smoke_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "quality_smoke.yaml"
    rubrics = load_task_rubrics(fixture)
    assert len(rubrics) == 3
    for r in rubrics:
        assert r is not None
        assert "mock" in r.contains_any


# ---------------------------------------------------------------------------
# compute_task_quality
# ---------------------------------------------------------------------------


def test_compute_task_quality_all_pass() -> None:
    rubrics = [TaskRubric(contains_any=["yes"])]
    report = compute_task_quality(["yes please", "yes sure"], prompt_count=1, rubrics=rubrics)
    assert isinstance(report, TaskQualityReport)
    assert report.task_quality_pass_rate == pytest.approx(1.0)
    assert report.task_quality_checked_count == 2


def test_compute_task_quality_some_fail() -> None:
    rubrics = [TaskRubric(contains_any=["yes"])]
    report = compute_task_quality(["yes", "no"], prompt_count=1, rubrics=rubrics)
    assert report.task_quality_pass_rate == pytest.approx(0.5)
    assert report.task_quality_checked_count == 2


def test_compute_task_quality_null_rubric_skipped() -> None:
    rubrics = [TaskRubric(contains_any=["yes"]), None]
    # texts: [prompt0→rubric0, prompt1→None, prompt0→rubric0, prompt1→None]
    texts = ["yes", "irrelevant", "yes", "also irrelevant"]
    report = compute_task_quality(texts, prompt_count=2, rubrics=rubrics)
    assert report.task_quality_checked_count == 2
    assert report.task_quality_pass_rate == pytest.approx(1.0)


def test_compute_task_quality_empty_texts() -> None:
    rubrics = [TaskRubric(contains_any=["yes"])]
    report = compute_task_quality([], prompt_count=1, rubrics=rubrics)
    assert report.task_quality_pass_rate == pytest.approx(1.0)
    assert report.task_quality_checked_count == 0


def test_compute_task_quality_rubrics_shorter_than_prompts() -> None:
    """Prompts beyond rubric list length are not checked."""
    rubrics = [TaskRubric(contains_all=["yes"])]
    # 3 prompts, only position 0 has a rubric
    texts = ["yes", "no", "no"]
    report = compute_task_quality(texts, prompt_count=3, rubrics=rubrics)
    assert report.task_quality_checked_count == 1
    assert report.task_quality_pass_rate == pytest.approx(1.0)


def test_compute_task_quality_cycling() -> None:
    """texts cycle through prompt indices; rubric at position 0 gets applied each cycle."""
    rubrics = [TaskRubric(contains_all=["ok"]), None]
    # 6 texts, 2 prompts → positions 0,1,0,1,0,1 → rubric applied for 0,2,4
    texts = ["ok", "x", "ok", "x", "fail", "x"]
    report = compute_task_quality(texts, prompt_count=2, rubrics=rubrics)
    assert report.task_quality_checked_count == 3
    assert report.task_quality_pass_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_runner_with_quality_file(tmp_path: Path) -> None:
    """run_benchmark populates task_quality fields when quality_file is configured."""
    from llm_inference_benchmark.backends.mock import MockBackend
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("Question 1\nQuestion 2\nQuestion 3\n")

    quality_file = tmp_path / "quality.yaml"
    quality_file.write_text(
        "- contains_any:\n    - mock\n  min_chars: 3\n"
        "- contains_any:\n    - mock\n  min_chars: 3\n"
        "- contains_any:\n    - mock\n  min_chars: 3\n"
    )

    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=10)
    cfg = BenchmarkConfig(
        requests=6,
        warmup_requests=0,
        prompts_file=str(prompts_file),
        quality_file=str(quality_file),
    )
    prompts = load_prompts(prompts_file)
    report = run_benchmark(backend, cfg, prompts)

    assert report.task_quality_pass_rate is not None
    assert report.task_quality_pass_rate == pytest.approx(1.0)
    assert report.task_quality_checked_count == 6


def test_runner_without_quality_file(tmp_path: Path) -> None:
    """task_quality fields are None when no quality_file is configured."""
    from llm_inference_benchmark.backends.mock import MockBackend
    from llm_inference_benchmark.config import BenchmarkConfig
    from llm_inference_benchmark.runner import load_prompts, run_benchmark

    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("Question 1\n")

    backend = MockBackend(model="test", latency_ms=0, tokens_per_response=5)
    cfg = BenchmarkConfig(requests=3, warmup_requests=0, prompts_file=str(prompts_file))
    prompts = load_prompts(prompts_file)
    report = run_benchmark(backend, cfg, prompts)

    assert report.task_quality_pass_rate is None
    assert report.task_quality_checked_count is None
