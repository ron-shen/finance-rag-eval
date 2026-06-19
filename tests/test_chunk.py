from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import chunk


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_write_chunks_jsonl_splits_paragraphs_and_preserves_page_metadata(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "pages.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    second_output_path = tmp_path / "chunks-again.jsonl"
    pages = [
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme Corp",
            "period": 2022,
            "page": 0,
            "text": "Revenue increased.\n\nMargins improved.",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
        },
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme Corp",
            "period": 2022,
            "page": 1,
            "text": "Cash flow was stable.",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
        },
    ]
    write_jsonl(input_path, pages)

    chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=output_path,
        strategy="paragraph",
        chunk_size=100,
        chunk_overlap=0,
    )
    chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=second_output_path,
        strategy="paragraph",
        chunk_size=100,
        chunk_overlap=0,
    )

    records = read_jsonl(output_path)
    assert [
        {key: value for key, value in record.items() if key != "chunk_id"}
        for record in records
    ] == [
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme Corp",
            "period": 2022,
            "page": 0,
            "text": "Revenue increased.",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
            "chunk_index": 0,
            "chunking_strategy": "paragraph",
        },
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme Corp",
            "period": 2022,
            "page": 0,
            "text": "Margins improved.",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
            "chunk_index": 1,
            "chunking_strategy": "paragraph",
        },
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme Corp",
            "period": 2022,
            "page": 1,
            "text": "Cash flow was stable.",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
            "chunk_index": 0,
            "chunking_strategy": "paragraph",
        },
    ]

    chunk_ids = [record["chunk_id"] for record in records]
    assert len(set(chunk_ids)) == len(chunk_ids)
    assert chunk_ids == [record["chunk_id"] for record in read_jsonl(second_output_path)]


def test_write_chunks_jsonl_supports_word_strategy(tmp_path: Path) -> None:
    input_path = tmp_path / "pages.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "doc_name": "BETA_2023_10Q",
                "company": "Beta Inc",
                "period": "Q2 2023",
                "page": 4,
                "text": "alpha beta gamma delta epsilon",
                "source_path": "pdfs/BETA_2023_10Q.pdf",
                "doc_type": "pdf",
            }
        ],
    )

    chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=output_path,
        strategy="word",
        chunk_size=2,
        chunk_overlap=0,
    )

    records = read_jsonl(output_path)
    assert [record["text"] for record in records] == [
        "alpha beta",
        "gamma delta",
        "epsilon",
    ]
    assert [record["chunk_index"] for record in records] == [0, 1, 2]
    assert {record["chunking_strategy"] for record in records} == {"word"}


def test_write_chunks_jsonl_supports_page_strategy_without_splitting(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "pages.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    second_output_path = tmp_path / "chunks-again.jsonl"
    pages = [
        {
            "doc_name": "GAMMA_2024_10K",
            "company": "Gamma LLC",
            "period": 2024,
            "page": 7,
            "text": "First page keeps every sentence.\n\n"
            "This paragraph also stays on the same chunk.",
            "source_path": "pdfs/GAMMA_2024_10K.pdf",
            "doc_type": "pdf",
            "section": "Management Discussion",
        },
        {
            "doc_name": "GAMMA_2024_10K",
            "company": "Gamma LLC",
            "period": 2024,
            "page": 8,
            "text": " ".join(f"token-{index}" for index in range(30)),
            "source_path": "pdfs/GAMMA_2024_10K.pdf",
            "doc_type": "pdf",
            "section": "Liquidity",
        },
    ]
    write_jsonl(input_path, pages)

    chunk_count = chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=output_path,
        strategy="page",
        chunk_size=20,
        chunk_overlap=0,
    )
    second_chunk_count = chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=second_output_path,
        strategy="page",
        chunk_size=20,
        chunk_overlap=0,
    )

    assert chunk_count == len(pages)
    assert second_chunk_count == len(pages)
    records = read_jsonl(output_path)
    assert [
        {key: value for key, value in record.items() if key != "chunk_id"}
        for record in records
    ] == [
        {
            **page,
            "chunk_index": 0,
            "chunking_strategy": "page",
        }
        for page in pages
    ]

    chunk_ids = [record["chunk_id"] for record in records]
    assert len(set(chunk_ids)) == len(chunk_ids)
    assert chunk_ids == [record["chunk_id"] for record in read_jsonl(second_output_path)]


def test_write_chunks_jsonl_supports_token_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "pages.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    second_output_path = tmp_path / "chunks-again.jsonl"
    splitter_inits: list[dict[str, int]] = []

    class FakeTokenTextSplitter:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            splitter_inits.append(
                {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            )

        def split_text(self, text: str) -> list[str]:
            tokens = text.split()
            step = self.chunk_size - self.chunk_overlap
            return [
                " ".join(tokens[start : start + self.chunk_size])
                for start in range(0, len(tokens), step)
                if tokens[start : start + self.chunk_size]
            ]

    monkeypatch.setattr(
        chunk,
        "TokenTextSplitter",
        FakeTokenTextSplitter,
        raising=False,
    )
    write_jsonl(
        input_path,
        [
            {
                "doc_name": "DELTA_2024_10Q",
                "company": "Delta Co",
                "period": "Q1 2024",
                "page": 3,
                "text": "alpha beta gamma delta epsilon zeta",
                "source_path": "pdfs/DELTA_2024_10Q.pdf",
                "doc_type": "pdf",
            }
        ],
    )

    chunk_count = chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=output_path,
        strategy="token",
        chunk_size=3,
        chunk_overlap=1,
    )
    second_chunk_count = chunk.write_chunks_jsonl(
        input_path=input_path,
        output_path=second_output_path,
        strategy="token",
        chunk_size=3,
        chunk_overlap=1,
    )

    assert chunk_count == 3
    assert second_chunk_count == 3
    assert splitter_inits == [
        {"chunk_size": 3, "chunk_overlap": 1},
        {"chunk_size": 3, "chunk_overlap": 1},
    ]

    records = read_jsonl(output_path)
    assert [
        {key: value for key, value in record.items() if key != "chunk_id"}
        for record in records
    ] == [
        {
            "doc_name": "DELTA_2024_10Q",
            "company": "Delta Co",
            "period": "Q1 2024",
            "page": 3,
            "text": "alpha beta gamma",
            "source_path": "pdfs/DELTA_2024_10Q.pdf",
            "doc_type": "pdf",
            "chunk_index": 0,
            "chunking_strategy": "token",
        },
        {
            "doc_name": "DELTA_2024_10Q",
            "company": "Delta Co",
            "period": "Q1 2024",
            "page": 3,
            "text": "gamma delta epsilon",
            "source_path": "pdfs/DELTA_2024_10Q.pdf",
            "doc_type": "pdf",
            "chunk_index": 1,
            "chunking_strategy": "token",
        },
        {
            "doc_name": "DELTA_2024_10Q",
            "company": "Delta Co",
            "period": "Q1 2024",
            "page": 3,
            "text": "epsilon zeta",
            "source_path": "pdfs/DELTA_2024_10Q.pdf",
            "doc_type": "pdf",
            "chunk_index": 2,
            "chunking_strategy": "token",
        },
    ]

    chunk_ids = [record["chunk_id"] for record in records]
    assert len(set(chunk_ids)) == len(chunk_ids)
    assert chunk_ids == [record["chunk_id"] for record in read_jsonl(second_output_path)]


def test_write_chunks_jsonl_reports_missing_token_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "pages.jsonl"
    output_path = tmp_path / "chunks.jsonl"

    class MissingTokenDependency:
        def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
            raise ImportError("missing tokenizer")

    monkeypatch.setattr(
        chunk,
        "TokenTextSplitter",
        MissingTokenDependency,
        raising=False,
    )
    write_jsonl(
        input_path,
        [
            {
                "doc_name": "DELTA_2024_10Q",
                "company": "Delta Co",
                "period": "Q1 2024",
                "page": 3,
                "text": "alpha beta gamma",
                "source_path": "pdfs/DELTA_2024_10Q.pdf",
                "doc_type": "pdf",
            }
        ],
    )

    with pytest.raises(ImportError, match="token strategy requires tiktoken"):
        chunk.write_chunks_jsonl(
            input_path=input_path,
            output_path=output_path,
            strategy="token",
            chunk_size=3,
            chunk_overlap=1,
        )

    assert not output_path.exists()


def test_write_chunks_jsonl_rejects_unsupported_strategy(tmp_path: Path) -> None:
    input_path = tmp_path / "pages.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "doc_name": "ACME_2022_10K",
                "company": "Acme Corp",
                "period": 2022,
                "page": 0,
                "text": "Revenue increased.",
                "source_path": "pdfs/ACME_2022_10K.pdf",
                "doc_type": "pdf",
            }
        ],
    )

    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        chunk.write_chunks_jsonl(
            input_path=input_path,
            output_path=output_path,
            strategy="heading",
            chunk_size=100,
            chunk_overlap=0,
        )

    assert not output_path.exists()
