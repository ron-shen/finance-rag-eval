from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


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


def test_generate_openai_answer_sends_explicit_json_output_example() -> None:
    answer = importlib.import_module("src.answer")
    calls: list[dict[str, Any]] = []
    generated = {
        "answer": "3M's FY2018 capital expenditures were $1.577 billion.",
        "citations": [{"doc_name": "3M_2018_10K", "page": 59}],
    }

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(generated))
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    retrieval = {
        "question": "What is the FY2018 capital expenditure amount for 3M?",
        "retrieved_chunks": [
            {
                "doc_name": "3M_2018_10K",
                "page": 59,
                "text": "Purchases of property, plant and equipment were 1,577.",
            }
        ],
    }

    result = answer.generate_openai_answer(
        retrieval,
        model="gpt-4.1-mini",
        client=fake_client,
    )

    assert result == generated
    assert len(calls) == 1
    messages = calls[0]["messages"]
    system_prompt = next(
        message["content"] for message in messages if message["role"] == "system"
    )
    assert (
        """{
  "answer": "3M's FY2018 capital expenditures were $1.577 billion.",
  "citations": [
    {"doc_name": "3M_2018_10K", "page": 59}
  ]
}"""
        in system_prompt
    )


def test_write_answers_jsonl_generates_answers_with_citations_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    answer = importlib.import_module("src.answer")
    retrievals_path = tmp_path / "retrievals.jsonl"
    output_path = tmp_path / "answers.jsonl"
    retrieval_record = {
        "financebench_id": "financebench_id_03029",
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question_type": "metrics-generated",
        "question": "What is the FY2018 capital expenditure amount for 3M?",
        "answer": "$1577.00",
        "retrieved_chunks": [
            {
                "doc_name": "3M_2018_10K",
                "page": 59,
                "score": 0.92,
                "text": "Purchases of property, plant and equipment were 1,577.",
            },
            {
                "doc_name": "3M_2018_10K",
                "page": 57,
                "score": 0.37,
                "text": "Property, plant and equipment net was 8,738.",
            },
        ],
    }
    write_jsonl(retrievals_path, [retrieval_record])
    generator_calls: list[dict[str, Any]] = []

    def fake_generator(
        retrieval: dict[str, Any],
        *,
        model: str,
    ) -> dict[str, Any]:
        generator_calls.append({"retrieval": retrieval, "model": model})
        return {
            "answer": "3M's FY2018 capital expenditures were $1.577 billion.",
            "citations": [{"doc_name": "3M_2018_10K", "page": 59}],
        }

    written_count = answer.write_answers_jsonl(
        retrievals_path=retrievals_path,
        output_path=output_path,
        generator=fake_generator,
        model="gpt-4.1-mini",
    )

    assert written_count == 1
    assert generator_calls == [
        {"retrieval": retrieval_record, "model": "gpt-4.1-mini"}
    ]
    output_rows = read_jsonl(output_path)
    answer_latency_ms = output_rows[0].pop("answer_latency_ms")
    assert (
        isinstance(answer_latency_ms, (int, float))
        and not isinstance(answer_latency_ms, bool)
        and answer_latency_ms >= 0
    )
    assert output_rows == [
        {
            **retrieval_record,
            "generated_answer": (
                "3M's FY2018 capital expenditures were $1.577 billion."
            ),
            "citations": [{"doc_name": "3M_2018_10K", "page": 59}],
            "answer_model": "gpt-4.1-mini",
        }
    ]


@pytest.mark.parametrize(
    "generation",
    [
        {"answer": "No citation field."},
        {"answer": "Empty citations.", "citations": []},
        {"answer": "Missing doc name.", "citations": [{"page": 59}]},
        {"answer": "Missing page.", "citations": [{"doc_name": "3M_2018_10K"}]},
        {"answer": "Blank doc name.", "citations": [{"doc_name": "", "page": 59}]},
        {
            "answer": "Null page.",
            "citations": [{"doc_name": "3M_2018_10K", "page": None}],
        },
    ],
)
def test_write_answers_jsonl_requires_citations_with_doc_name_and_page(
    tmp_path: Path,
    generation: dict[str, Any],
) -> None:
    answer = importlib.import_module("src.answer")
    retrievals_path = tmp_path / "retrievals.jsonl"
    output_path = tmp_path / "answers.jsonl"
    write_jsonl(
        retrievals_path,
        [
            {
                "financebench_id": "financebench_id_03029",
                "question": "What is the FY2018 capital expenditure amount for 3M?",
                "answer": "$1577.00",
                "retrieved_chunks": [
                    {
                        "doc_name": "3M_2018_10K",
                        "page": 59,
                        "score": 0.92,
                        "text": "Purchases of property, plant and equipment were 1,577.",
                    }
                ],
            }
        ],
    )

    def fake_generator(
        retrieval: dict[str, Any],
        *,
        model: str,
    ) -> dict[str, Any]:
        return generation

    with pytest.raises(ValueError, match="citations.*doc_name.*page"):
        answer.write_answers_jsonl(
            retrievals_path=retrievals_path,
            output_path=output_path,
            generator=fake_generator,
            model="gpt-4.1-mini",
        )

    assert not output_path.exists()
