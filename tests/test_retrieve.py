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


def test_infer_chroma_metadata_filter_resolves_exact_doc_from_chunk_catalog() -> None:
    retrieve = importlib.import_module("src.retrieve")
    chunks = [
        {
            "doc_name": "3M_2018_10K",
            "company": "3M",
            "period": 2018,
            "page": 1,
            "text": "2018 revenue discussion.",
        },
        {
            "doc_name": "3M_2018_10K",
            "company": "3M",
            "period": 2018,
            "page": 2,
            "text": "More 2018 filing text.",
        },
        {
            "doc_name": "3M_2019_10K",
            "company": "3M",
            "period": 2019,
            "page": 1,
            "text": "2019 revenue discussion.",
        },
    ]

    catalog = retrieve.build_document_catalog(chunks)

    assert retrieve.infer_chroma_metadata_filter(
        "What is the revenue of 3M in 2018?",
        catalog,
    ) == {"doc_name": "3M_2018_10K"}


def test_infer_chroma_metadata_filter_falls_back_to_company_period_filter() -> None:
    retrieve = importlib.import_module("src.retrieve")
    chunks = [
        {
            "doc_name": "3M_2018_annual_report",
            "company": "3M",
            "period": 2018,
            "page": 1,
            "text": "2018 annual report text.",
        },
        {
            "doc_name": "3M_2018Q4_EARNINGS",
            "company": "3M",
            "period": 2018,
            "page": 1,
            "text": "2018 earnings text.",
        },
        {
            "doc_name": "3M_2019_10K",
            "company": "3M",
            "period": 2019,
            "page": 1,
            "text": "2019 filing text.",
        },
    ]

    catalog = retrieve.build_document_catalog(chunks)

    assert retrieve.infer_chroma_metadata_filter(
        "What is the revenue of 3M in 2018?",
        catalog,
    ) == {"$and": [{"company": "3M"}, {"period": 2018}]}


def test_infer_chroma_metadata_filter_returns_none_without_metadata_match() -> None:
    retrieve = importlib.import_module("src.retrieve")
    chunks = [
        {
            "doc_name": "3M_2018_10K",
            "company": "3M",
            "period": 2018,
            "page": 1,
            "text": "2018 filing text.",
        },
    ]

    catalog = retrieve.build_document_catalog(chunks)

    assert (
        retrieve.infer_chroma_metadata_filter(
            "What was the capital expenditure amount?",
            catalog,
        )
        is None
    )


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


def test_write_chroma_retrievals_jsonl_infers_metadata_filter_from_catalog(
    tmp_path: Path,
) -> None:
    retrieve = importlib.import_module("src.retrieve")
    questions_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "retrievals.jsonl"
    chunks = [
        {
            "doc_name": "3M_2018_10K",
            "company": "3M",
            "period": 2018,
            "page": 42,
            "text": "3M 2018 revenue discussion.",
        },
        {
            "doc_name": "3M_2019_10K",
            "company": "3M",
            "period": 2019,
            "page": 43,
            "text": "3M 2019 revenue discussion.",
        },
    ]
    question = {
        "financebench_id": "financebench_id_3m_revenue_2018",
        "question": "What is the revenue of 3M in 2018?",
    }

    class FakeEmbeddingDatum:
        embedding = [0.1, 0.2, 0.3]

    class FakeEmbeddingResponse:
        data = [FakeEmbeddingDatum()]

    class FakeEmbeddingClient:
        def __init__(self) -> None:
            self.embeddings = self

        def create(self, *, model: str, input: list[str]) -> FakeEmbeddingResponse:
            assert model == "text-embedding-3-small"
            assert input == [question["question"]]
            return FakeEmbeddingResponse()

    class FakeCollection:
        def __init__(self) -> None:
            self.where_calls: list[dict[str, Any] | None] = []

        def query(
            self,
            *,
            query_embeddings: list[list[float]],
            n_results: int,
            include: list[str],
            where: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            assert query_embeddings == [[0.1, 0.2, 0.3]]
            assert n_results == 1
            assert include == ["documents", "metadatas", "distances"]
            self.where_calls.append(where)
            return {
                "documents": [["3M reported 2018 revenue."]],
                "metadatas": [[{"doc_name": "3M_2018_10K", "page": 42}]],
                "distances": [[0.0]],
            }

    collection = FakeCollection()
    write_jsonl(questions_path, [question])

    written_count = retrieve.write_chroma_retrievals_jsonl(
        questions_path=questions_path,
        output_path=output_path,
        collection=collection,
        embedding_client=FakeEmbeddingClient(),
        model="text-embedding-3-small",
        top_k=1,
        document_catalog=retrieve.build_document_catalog(chunks),
    )

    assert written_count == 1
    assert collection.where_calls == [{"doc_name": "3M_2018_10K"}]
