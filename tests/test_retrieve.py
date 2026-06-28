from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


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


def test_write_retrievals_jsonl_ranks_top_k_chunks_and_preserves_question_metadata(
    tmp_path: Path,
) -> None:
    retrieve = importlib.import_module("src.retrieve")
    questions_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "retrievals.jsonl"
    question = {
        "financebench_id": "financebench_id_03029",
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question_type": "metrics-generated",
        "question": "What is the FY2018 capital expenditure amount for 3M?",
        "answer": "$1577.00",
    }
    chunks = [
        {
            "chunk_id": "75a75dacb50db219383b316e0909ae929813b5560fc9f8e006e0e8c522117287",
            "doc_name": "3M_2018_10K",
            "page": 59,
            "text": "Purchases of property, plant and equipment were 1,577.",
            "chunk_index": 3,
        },
        {
            "chunk_id": "517ebdc09faf1af342da45adc5ffa07f028f887a565c637f5e053c1d59dc9cdf",
            "doc_name": "3M_2018_10K",
            "page": 57,
            "text": "Property, plant and equipment net was 8,738.",
            "chunk_index": 1,
        },
        {
            "chunk_id": "8f5082ae18373377ce4ca1b39add70b44d8d9149760d4c0ed20d37eddcbbb2e4",
            "doc_name": "3M_2018_10K",
            "page": 12,
            "text": "Risk factors discussion.",
            "chunk_index": 0,
        },
    ]
    scores_by_page = {
        59: 0.92,
        57: 0.37,
        12: 0.08,
    }
    scorer_calls: list[tuple[str, str]] = []

    def fake_scorer(
        question_record: dict[str, Any],
        chunk_record: dict[str, Any],
    ) -> float:
        scorer_calls.append(
            (question_record["financebench_id"], chunk_record["chunk_id"])
        )
        return scores_by_page[chunk_record["page"]]

    write_jsonl(questions_path, [question])
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    written_count = retrieve.write_retrievals_jsonl(
        questions_path=questions_path,
        chunks=chunks,
        output_path=output_path,
        scorer=fake_scorer,
        top_k=2,
    )

    assert written_count == 1
    assert scorer_calls == [
        ("financebench_id_03029", chunk_ids[0]),
        ("financebench_id_03029", chunk_ids[1]),
        ("financebench_id_03029", chunk_ids[2]),
    ]
    assert read_jsonl(output_path) == [
        {
            **question,
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
    ]
