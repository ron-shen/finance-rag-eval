from __future__ import annotations

import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest


HEURISTIC_METRIC_KEYS = {
    "doc_hit_at_k",
    "evidence_page_hit_at_k",
    "mrr",
    "numeric_answer_match",
    "citation_accuracy",
}

RAGAS_METRIC_KEYS = (
    "context_precision",
    "context_recall",
    "response_relevancy",
    "faithfulness",
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ragas_metric_subset(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in RAGAS_METRIC_KEYS}


def test_write_metrics_jsonl_uses_injected_ragas_evaluator_and_preserves_rows(
    tmp_path: Path,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_rows = [
        {
            "financebench_id": "financebench_id_03029",
            "company": "3M",
            "doc_name": "3M_2018_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Information extraction",
            "domain_question_num": None,
            "question": "What was 3M's FY2018 capital expenditure?",
            "answer": "$1577.00",
            "generated_answer": "3M's FY2018 capital expenditure was $1.577 billion.",
            "retrieved_chunks": [
                {
                    "doc_name": "3M_2018_10K",
                    "page": 59,
                    "text": "Purchases of property, plant and equipment were 1,577.",
                },
                {
                    "doc_name": "3M_2018_10K",
                    "page": 57,
                    "text": "Property, plant and equipment net was 8,738.",
                },
            ],
        },
        {
            "financebench_id": "financebench_id_04672",
            "company": "3M",
            "doc_name": "3M_2018_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Information extraction",
            "domain_question_num": 2,
            "question": "What was 3M's FY2018 net PP&E?",
            "answer": "$8.70 billion",
            "generated_answer": "3M's FY2018 net PP&E was $8.738 billion.",
            "retrieved_chunks": [
                {
                    "doc_name": "3M_2018_10K",
                    "page": 57,
                    "text": "Property, plant and equipment net was 8,738.",
                }
            ],
        },
    ]
    write_jsonl(input_path, input_rows)
    evaluator_calls: list[dict[str, Any]] = []

    class FakeRagasEvaluator:
        def evaluate(
            self,
            samples: list[dict[str, Any]],
            *,
            metrics: tuple[str, ...],
        ) -> list[dict[str, float]]:
            evaluator_calls.append({"samples": samples, "metrics": metrics})
            return [
                {
                    "context_precision": 0.91,
                    "context_recall": 0.82,
                    "response_relevancy": 0.73,
                    "faithfulness": 0.64,
                },
                {
                    "context_precision": 0.55,
                    "context_recall": 0.46,
                    "response_relevancy": 0.37,
                    "faithfulness": 0.28,
                },
            ]

    written_count = evaluate.write_metrics_jsonl(
        input_path=input_path,
        output_path=output_path,
        top_k=1,
        evaluator=FakeRagasEvaluator(),
    )

    assert written_count == 2
    assert evaluator_calls == [
        {
            "metrics": RAGAS_METRIC_KEYS,
            "samples": [
                {
                    "question": "What was 3M's FY2018 capital expenditure?",
                    "answer": (
                        "3M's FY2018 capital expenditure was $1.577 billion."
                    ),
                    "contexts": [
                        "Purchases of property, plant and equipment were 1,577."
                    ],
                    "ground_truth": "$1577.00",
                },
                {
                    "question": "What was 3M's FY2018 net PP&E?",
                    "answer": "3M's FY2018 net PP&E was $8.738 billion.",
                    "contexts": [
                        "Property, plant and equipment net was 8,738.",
                    ],
                    "ground_truth": "$8.70 billion",
                },
            ],
        }
    ]
    output_rows = read_jsonl(output_path)
    assert [row["financebench_id"] for row in output_rows] == [
        "financebench_id_03029",
        "financebench_id_04672",
    ]
    for input_row, output_row in zip(input_rows, output_rows):
        for field in evaluate.IDENTITY_FIELDS:
            assert output_row[field] == input_row[field]
    assert [ragas_metric_subset(row) for row in output_rows] == [
        {
            "context_precision": pytest.approx(0.91),
            "context_recall": pytest.approx(0.82),
            "response_relevancy": pytest.approx(0.73),
            "faithfulness": pytest.approx(0.64),
        },
        {
            "context_precision": pytest.approx(0.55),
            "context_recall": pytest.approx(0.46),
            "response_relevancy": pytest.approx(0.37),
            "faithfulness": pytest.approx(0.28),
        },
    ]
    for output_row in output_rows:
        assert HEURISTIC_METRIC_KEYS.isdisjoint(output_row)


def test_write_metrics_jsonl_uses_default_ragas_evaluator_when_not_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_rows = [
        {
            "financebench_id": "row-default",
            "company": "Acme",
            "doc_name": "ACME_2023_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Information extraction",
            "domain_question_num": 1,
            "question": "What was Acme's FY2023 revenue?",
            "answer": "$10.0 million",
            "generated_answer": "Acme's FY2023 revenue was $10.0 million.",
            "retrieved_chunks": [
                {
                    "doc_name": "ACME_2023_10K",
                    "page": 12,
                    "text": "Revenue was $10.0 million in FY2023.",
                },
                {
                    "doc_name": "ACME_2023_10K",
                    "page": 13,
                    "text": "Operating income was $2.0 million.",
                },
            ],
        }
    ]
    write_jsonl(input_path, input_rows)
    evaluator_events: list[dict[str, Any]] = []

    class FakeDefaultRagasEvaluator:
        def __init__(self, **kwargs: Any) -> None:
            evaluator_events.append({"event": "init", "kwargs": kwargs})

        def evaluate(
            self,
            samples: list[dict[str, Any]],
            *,
            metrics: tuple[str, ...],
        ) -> list[dict[str, float]]:
            evaluator_events.append(
                {
                    "event": "evaluate",
                    "samples": samples,
                    "metrics": metrics,
                }
            )
            return [
                {
                    "context_precision": 0.88,
                    "context_recall": 0.77,
                    "response_relevancy": 0.66,
                    "faithfulness": 0.55,
                }
            ]

    monkeypatch.setattr(evaluate, "DefaultRagasEvaluator", FakeDefaultRagasEvaluator)

    written_count = evaluate.write_metrics_jsonl(
        input_path=input_path,
        output_path=output_path,
        top_k=2,
    )

    assert written_count == 1
    assert evaluator_events == [
        {"event": "init", "kwargs": {}},
        {
            "event": "evaluate",
            "metrics": RAGAS_METRIC_KEYS,
            "samples": [
                {
                    "question": "What was Acme's FY2023 revenue?",
                    "answer": "Acme's FY2023 revenue was $10.0 million.",
                    "contexts": [
                        "Revenue was $10.0 million in FY2023.",
                        "Operating income was $2.0 million.",
                    ],
                    "ground_truth": "$10.0 million",
                }
            ],
        },
    ]
    output_rows = read_jsonl(output_path)
    assert [row["financebench_id"] for row in output_rows] == ["row-default"]
    assert [ragas_metric_subset(row) for row in output_rows] == [
        {
            "context_precision": pytest.approx(0.88),
            "context_recall": pytest.approx(0.77),
            "response_relevancy": pytest.approx(0.66),
            "faithfulness": pytest.approx(0.55),
        }
    ]
    assert HEURISTIC_METRIC_KEYS.isdisjoint(output_rows[0])


def test_write_metrics_jsonl_public_api_has_no_use_ragas_flag() -> None:
    evaluate = importlib.import_module("src.evaluate")

    parameters = inspect.signature(evaluate.write_metrics_jsonl).parameters

    assert "use_ragas" not in parameters


def test_parse_args_no_longer_exposes_ragas_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    args = evaluate.parse_args()

    assert not hasattr(args, "ragas")


def test_default_ragas_evaluator_uses_ragas_compatible_llm_by_default() -> None:
    evaluate = importlib.import_module("src.evaluate")

    evaluator = evaluate.DefaultRagasEvaluator()

    assert evaluator.llm_model == "gpt-4o-mini"
    assert evaluator.embedding_model == "text-embedding-3-small"


def test_cli_default_ragas_evaluator_keeps_ragas_compatible_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    args = evaluate.parse_args()
    evaluator = evaluate._default_ragas_evaluator(args)

    assert evaluator.llm_model == "gpt-4o-mini"


def test_cli_ragas_llm_model_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--ragas-llm-model",
            "custom-ragas-model",
        ],
    )

    args = evaluate.parse_args()
    evaluator = evaluate._default_ragas_evaluator(args)

    assert evaluator.llm_model == "custom-ragas-model"


def test_main_uses_default_ragas_evaluator_without_ragas_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    input_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "financebench_id": "row-cli",
                "company": "Acme",
                "doc_name": "ACME_2023_10K",
                "question_type": "metrics-generated",
                "question_reasoning": "Information extraction",
                "domain_question_num": None,
                "question": "What was Acme's FY2023 cash flow?",
                "answer": "$3.0 million",
                "generated_answer": "Acme's FY2023 cash flow was $3.0 million.",
                "retrieved_chunks": [
                    {
                        "doc_name": "ACME_2023_10K",
                        "page": 7,
                        "text": "Cash flow was $3.0 million in FY2023.",
                    }
                ],
            }
        ],
    )
    evaluator_calls: list[dict[str, Any]] = []

    class FakeDefaultRagasEvaluator:
        def __init__(self, **kwargs: Any) -> None:
            evaluator_calls.append({"event": "init", "kwargs": kwargs})

        def evaluate(
            self,
            samples: list[dict[str, Any]],
            *,
            metrics: tuple[str, ...],
        ) -> list[dict[str, float]]:
            evaluator_calls.append(
                {
                    "event": "evaluate",
                    "samples": samples,
                    "metrics": metrics,
                }
            )
            return [
                {
                    "context_precision": 0.44,
                    "context_recall": 0.33,
                    "response_relevancy": 0.22,
                    "faithfulness": 0.11,
                }
            ]

    monkeypatch.setattr(evaluate, "DefaultRagasEvaluator", FakeDefaultRagasEvaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--top-k",
            "1",
        ],
    )

    evaluate.main()

    assert evaluator_calls == [
        {"event": "init", "kwargs": {}},
        {
            "event": "evaluate",
            "metrics": RAGAS_METRIC_KEYS,
            "samples": [
                {
                    "question": "What was Acme's FY2023 cash flow?",
                    "answer": "Acme's FY2023 cash flow was $3.0 million.",
                    "contexts": ["Cash flow was $3.0 million in FY2023."],
                    "ground_truth": "$3.0 million",
                }
            ],
        },
    ]
    assert ragas_metric_subset(read_jsonl(output_path)[0]) == {
        "context_precision": pytest.approx(0.44),
        "context_recall": pytest.approx(0.33),
        "response_relevancy": pytest.approx(0.22),
        "faithfulness": pytest.approx(0.11),
    }
    assert "Wrote metrics for 1 answers" in capsys.readouterr().out


def test_default_ragas_evaluator_uses_async_openai_client_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    captured: dict[str, Any] = {"llm_clients": [], "embedding_clients": []}

    class FakeSyncOpenAI:
        pass

    class FakeAsyncOpenAI:
        pass

    class FakeCollectionMetric:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def batch_score(self, inputs: list[dict[str, Any]]) -> list[float]:
            return [0.5 for _ in inputs]

    def fake_llm_factory(model: str, **kwargs: Any) -> dict[str, Any]:
        captured["llm_clients"].append(kwargs["client"])
        return {"model": model, **kwargs}

    def fake_embedding_factory(**kwargs: Any) -> dict[str, Any]:
        captured["embedding_clients"].append(kwargs["client"])
        return kwargs

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = FakeSyncOpenAI
    openai_module.AsyncOpenAI = FakeAsyncOpenAI

    ragas_module = types.ModuleType("ragas")
    ragas_module.__path__ = []

    embeddings_package = types.ModuleType("ragas.embeddings")
    embeddings_package.__path__ = []
    embeddings_base_module = types.ModuleType("ragas.embeddings.base")
    embeddings_base_module.embedding_factory = fake_embedding_factory

    llms_module = types.ModuleType("ragas.llms")
    llms_module.llm_factory = fake_llm_factory

    metrics_module = types.ModuleType("ragas.metrics")
    metrics_module.__path__ = []

    collections_module = types.ModuleType("ragas.metrics.collections")
    collections_module.AnswerRelevancy = FakeCollectionMetric
    collections_module.ContextPrecision = FakeCollectionMetric
    collections_module.ContextRecall = FakeCollectionMetric
    collections_module.Faithfulness = FakeCollectionMetric

    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", embeddings_package)
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", embeddings_base_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections_module)

    result = evaluate.DefaultRagasEvaluator(
        llm_model="test-llm",
        embedding_model="test-embedding",
    ).evaluate(
        [
            {
                "question": "What was Acme's FY2023 revenue?",
                "answer": "Acme's FY2023 revenue was $10.0 million.",
                "contexts": ["Revenue was $10.0 million in FY2023."],
                "ground_truth": "$10.0 million",
            }
        ],
        metrics=("context_precision", "response_relevancy"),
    )

    assert result == [
        {
            "context_precision": pytest.approx(0.5),
            "response_relevancy": pytest.approx(0.5),
        }
    ]
    assert len(captured["llm_clients"]) == 1
    assert len(captured["embedding_clients"]) == 1
    assert isinstance(captured["llm_clients"][0], FakeAsyncOpenAI)
    assert isinstance(captured["embedding_clients"][0], FakeAsyncOpenAI)
    assert captured["llm_clients"][0] is captured["embedding_clients"][0]


def test_default_ragas_evaluator_passes_openai_timeout_and_max_retries_to_async_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    created_clients: list[Any] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            created_clients.append(self)

    class FakeCollectionMetric:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def batch_score(self, inputs: list[dict[str, Any]]) -> list[float]:
            return [0.5 for _ in inputs]

    def fake_llm_factory(model: str, **kwargs: Any) -> dict[str, Any]:
        return {"model": model, **kwargs}

    def fake_embedding_factory(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    openai_module = types.ModuleType("openai")
    openai_module.AsyncOpenAI = FakeAsyncOpenAI

    ragas_module = types.ModuleType("ragas")
    ragas_module.__path__ = []

    embeddings_package = types.ModuleType("ragas.embeddings")
    embeddings_package.__path__ = []
    embeddings_base_module = types.ModuleType("ragas.embeddings.base")
    embeddings_base_module.embedding_factory = fake_embedding_factory

    llms_module = types.ModuleType("ragas.llms")
    llms_module.llm_factory = fake_llm_factory

    metrics_module = types.ModuleType("ragas.metrics")
    metrics_module.__path__ = []

    collections_module = types.ModuleType("ragas.metrics.collections")
    collections_module.AnswerRelevancy = FakeCollectionMetric
    collections_module.ContextPrecision = FakeCollectionMetric
    collections_module.ContextRecall = FakeCollectionMetric
    collections_module.Faithfulness = FakeCollectionMetric

    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", embeddings_package)
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", embeddings_base_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections_module)

    result = evaluate.DefaultRagasEvaluator(
        openai_timeout=45.5,
        openai_max_retries=6,
    ).evaluate(
        [
            {
                "question": "What was Acme's FY2023 revenue?",
                "answer": "Acme's FY2023 revenue was $10.0 million.",
                "contexts": ["Revenue was $10.0 million in FY2023."],
                "ground_truth": "$10.0 million",
            }
        ],
        metrics=("context_precision",),
    )

    assert result == [{"context_precision": pytest.approx(0.5)}]
    assert len(created_clients) == 1
    assert created_clients[0].kwargs == {
        "timeout": pytest.approx(45.5),
        "max_retries": 6,
    }


def test_parse_args_accepts_openai_timeout_and_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--openai-timeout",
            "45.5",
            "--openai-max-retries",
            "6",
        ],
    )

    args = evaluate.parse_args()

    assert args.openai_timeout == pytest.approx(45.5)
    assert args.openai_max_retries == 6


def test_default_ragas_evaluator_factory_forwards_openai_timeout_and_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    evaluator_calls: list[dict[str, Any]] = []

    class FakeDefaultRagasEvaluator:
        def __init__(self, **kwargs: Any) -> None:
            evaluator_calls.append(kwargs)

    monkeypatch.setattr(evaluate, "DefaultRagasEvaluator", FakeDefaultRagasEvaluator)

    evaluator = evaluate._default_ragas_evaluator(
        types.SimpleNamespace(
            ragas_llm_model=None,
            ragas_embedding_model=None,
            openai_timeout=45.5,
            openai_max_retries=6,
        )
    )

    assert isinstance(evaluator, FakeDefaultRagasEvaluator)
    assert evaluator_calls == [
        {
            "openai_timeout": pytest.approx(45.5),
            "openai_max_retries": 6,
        }
    ]


def test_score_collection_metric_prefers_score_over_batch_score() -> None:
    evaluate = importlib.import_module("src.evaluate")
    score_calls: list[dict[str, Any]] = []

    class FakeCollectionMetric:
        def score(self, **kwargs: Any) -> float:
            score_calls.append(kwargs)
            return float(len(score_calls))

        def batch_score(self, inputs: list[dict[str, Any]]) -> list[float]:
            pytest.fail("batch_score should not be used when score is available.")

    metric_inputs = [
        {
            "user_input": "What was Acme's FY2023 revenue?",
            "retrieved_contexts": ["Revenue was $10.0 million in FY2023."],
            "reference": "$10.0 million",
        },
        {
            "user_input": "What was Acme's FY2023 operating income?",
            "retrieved_contexts": ["Operating income was $2.0 million in FY2023."],
            "reference": "$2.0 million",
        },
    ]

    result = evaluate._score_collection_metric(
        FakeCollectionMetric(),
        metric_inputs,
        "context_precision",
    )

    assert result == [pytest.approx(1.0), pytest.approx(2.0)]
    assert score_calls == metric_inputs


def test_default_ragas_evaluator_scores_collection_metrics_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = importlib.import_module("src.evaluate")
    captured: dict[str, Any] = {"created": [], "batch_calls": []}

    class MetricResult:
        def __init__(self, value: float) -> None:
            self.value = value

    class FakeCollectionMetric:
        metric_name = "base"
        score_value = 0.0

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            captured["created"].append(
                {
                    "metric": self.metric_name,
                    "kwargs": kwargs,
                }
            )

        def batch_score(
            self,
            inputs: list[dict[str, Any]],
        ) -> list[MetricResult]:
            captured["batch_calls"].append(
                {
                    "metric": self.metric_name,
                    "inputs": inputs,
                }
            )
            return [MetricResult(self.score_value) for _ in inputs]

    class FakeContextPrecision(FakeCollectionMetric):
        metric_name = "context_precision"
        score_value = 0.91

    class FakeContextRecall(FakeCollectionMetric):
        metric_name = "context_recall"
        score_value = 0.82

    class FakeAnswerRelevancy(FakeCollectionMetric):
        metric_name = "answer_relevancy"
        score_value = 0.73

    class FakeFaithfulness(FakeCollectionMetric):
        metric_name = "faithfulness"
        score_value = 0.64

    def fake_llm_factory(model: str, **kwargs: Any) -> dict[str, Any]:
        return {"model": model, **kwargs}

    def fake_embedding_factory(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    class FakeOpenAI:
        pass

    class FakeAsyncOpenAI:
        def __init__(self) -> None:
            pytest.fail("Injected client should be used without creating a default client.")

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI
    openai_module.AsyncOpenAI = FakeAsyncOpenAI

    ragas_module = types.ModuleType("ragas")
    ragas_module.__path__ = []
    ragas_module.evaluate = pytest.fail

    embeddings_package = types.ModuleType("ragas.embeddings")
    embeddings_package.__path__ = []
    embeddings_base_module = types.ModuleType("ragas.embeddings.base")
    embeddings_base_module.embedding_factory = fake_embedding_factory

    llms_module = types.ModuleType("ragas.llms")
    llms_module.llm_factory = fake_llm_factory

    metrics_module = types.ModuleType("ragas.metrics")
    metrics_module.__path__ = []

    collections_module = types.ModuleType("ragas.metrics.collections")
    collections_module.AnswerRelevancy = FakeAnswerRelevancy
    collections_module.ContextPrecision = FakeContextPrecision
    collections_module.ContextRecall = FakeContextRecall
    collections_module.Faithfulness = FakeFaithfulness

    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", embeddings_package)
    monkeypatch.setitem(sys.modules, "ragas.embeddings.base", embeddings_base_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections_module)

    client = object()
    result = evaluate.DefaultRagasEvaluator(
        llm_model="test-llm",
        embedding_model="test-embedding",
        client=client,
    ).evaluate(
        [
            {
                "question": "What was Acme's FY2023 revenue?",
                "answer": "Acme's FY2023 revenue was $10.0 million.",
                "contexts": ["Revenue was $10.0 million in FY2023."],
                "ground_truth": "$10.0 million",
            }
        ],
        metrics=RAGAS_METRIC_KEYS,
    )

    assert result == [
        {
            "context_precision": pytest.approx(0.91),
            "context_recall": pytest.approx(0.82),
            "response_relevancy": pytest.approx(0.73),
            "faithfulness": pytest.approx(0.64),
        }
    ]
    assert captured["created"] == [
        {
            "metric": "context_precision",
            "kwargs": {"llm": {"model": "test-llm", "client": client}},
        },
        {
            "metric": "context_recall",
            "kwargs": {"llm": {"model": "test-llm", "client": client}},
        },
        {
            "metric": "answer_relevancy",
            "kwargs": {
                "llm": {"model": "test-llm", "client": client},
                "embeddings": {
                    "provider": "openai",
                    "model": "test-embedding",
                    "client": client,
                    "interface": "modern",
                },
            },
        },
        {
            "metric": "faithfulness",
            "kwargs": {"llm": {"model": "test-llm", "client": client}},
        },
    ]
    assert captured["batch_calls"] == [
        {
            "metric": "context_precision",
            "inputs": [
                {
                    "user_input": "What was Acme's FY2023 revenue?",
                    "retrieved_contexts": ["Revenue was $10.0 million in FY2023."],
                    "reference": "$10.0 million",
                }
            ],
        },
        {
            "metric": "context_recall",
            "inputs": [
                {
                    "user_input": "What was Acme's FY2023 revenue?",
                    "retrieved_contexts": ["Revenue was $10.0 million in FY2023."],
                    "reference": "$10.0 million",
                }
            ],
        },
        {
            "metric": "answer_relevancy",
            "inputs": [
                {
                    "user_input": "What was Acme's FY2023 revenue?",
                    "response": "Acme's FY2023 revenue was $10.0 million.",
                }
            ],
        },
        {
            "metric": "faithfulness",
            "inputs": [
                {
                    "user_input": "What was Acme's FY2023 revenue?",
                    "response": "Acme's FY2023 revenue was $10.0 million.",
                    "retrieved_contexts": ["Revenue was $10.0 million in FY2023."],
                }
            ],
        },
    ]
