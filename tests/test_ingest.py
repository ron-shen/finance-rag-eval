from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import ingest


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"doc_name": "A"}\n\n{"doc_name": "B"}\n', encoding="utf-8")

    assert ingest.read_jsonl(path) == [{"doc_name": "A"}, {"doc_name": "B"}]


def test_read_jsonl_reports_invalid_line_number(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"doc_name": "A"}\n{not valid json}\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        ingest.read_jsonl(path)

    assert str(exc_info.value) == f"Invalid JSON on {path}:2"


@pytest.mark.parametrize(
    ("doc_name", "expected"),
    [
        ("APPLE_2023Q3_10Q", 2023),
        ("MICROSOFT_2015_10K", 2015),
        ("NO_YEAR_INCLUDED", None),
    ],
)
def test_infer_period(doc_name: str, expected: int | None) -> None:
    assert ingest.infer_period(doc_name) == expected


def test_extract_pdf_pages_builds_page_records(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = Path("raw") / "pdfs" / "ACME_2022_10K.pdf"

    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class FakePdfReader:
        def __init__(self, path: str) -> None:
            assert path == str(pdf_path)
            self.pages = [FakePage("first page text"), FakePage(None)]

    monkeypatch.setattr(ingest, "PdfReader", FakePdfReader)

    doc_name, lines, page_count = ingest.extract_pdf_pages(
        {
            "doc_name": "ACME_2022_10K",
            "pdf_path": pdf_path,
            "company": "Acme",
            "period": 2022,
            "source_path": "pdfs/ACME_2022_10K.pdf",
        }
    )

    assert doc_name == "ACME_2022_10K"
    assert page_count == 2
    assert [json.loads(line) for line in lines] == [
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme",
            "period": 2022,
            "page": 0,
            "text": "first page text",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
        },
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme",
            "period": 2022,
            "page": 1,
            "text": "",
            "source_path": "pdfs/ACME_2022_10K.pdf",
            "doc_type": "pdf",
        },
    ]


def test_write_pages_jsonl_builds_tasks_in_doc_name_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    pdf_dir = raw_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    for doc_name in ("BETA_2020_10K", "ALPHA_2019_10K"):
        (pdf_dir / f"{doc_name}.pdf").touch()

    benchmark_path = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark_path,
        [
            {"doc_name": "BETA_2020_10K", "company": "Benchmark Beta"},
            {"doc_name": "ALPHA_2019_10K", "company": "Benchmark Alpha"},
            {"doc_name": "BETA_2020_10K", "company": "Duplicate Beta"},
        ],
    )
    document_info_path = tmp_path / "document_info.jsonl"
    write_jsonl(
        document_info_path,
        [
            {
                "doc_name": "BETA_2020_10K",
                "company": "Info Beta",
                "doc_period": "FY2020",
            },
            {"doc_name": "", "company": "Ignored"},
        ],
    )

    seen_tasks: list[dict] = []

    def fake_extract_pdf_pages(task: dict) -> tuple[str, list[str], int]:
        seen_tasks.append(task)
        return (
            task["doc_name"],
            [json.dumps({"doc_name": task["doc_name"], "page": 0})],
            1,
        )

    monkeypatch.setattr(ingest, "extract_pdf_pages", fake_extract_pdf_pages)

    output_path = tmp_path / "processed" / "pages.jsonl"
    doc_count, page_count = ingest.write_pages_jsonl(
        benchmark_path=benchmark_path,
        document_info_path=document_info_path,
        raw_dir=raw_dir,
        pdf_dir=pdf_dir,
        output_path=output_path,
        workers=1,
    )

    assert (doc_count, page_count) == (2, 2)
    assert seen_tasks == [
        {
            "doc_name": "ALPHA_2019_10K",
            "pdf_path": str(pdf_dir / "ALPHA_2019_10K.pdf"),
            "company": "Benchmark Alpha",
            "period": 2019,
            "source_path": "pdfs/ALPHA_2019_10K.pdf",
        },
        {
            "doc_name": "BETA_2020_10K",
            "pdf_path": str(pdf_dir / "BETA_2020_10K.pdf"),
            "company": "Info Beta",
            "period": "FY2020",
            "source_path": "pdfs/BETA_2020_10K.pdf",
        },
    ]
    output_records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output_records == [
        {"doc_name": "ALPHA_2019_10K", "page": 0},
        {"doc_name": "BETA_2020_10K", "page": 0},
    ]


def test_write_pages_jsonl_raises_for_missing_referenced_pdfs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    pdf_dir = raw_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "PRESENT_2021_10K.pdf").touch()

    benchmark_path = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark_path,
        [
            {"doc_name": "PRESENT_2021_10K", "company": "Present"},
            {"doc_name": "MISSING_2022_10K", "company": "Missing"},
        ],
    )
    document_info_path = tmp_path / "document_info.jsonl"
    write_jsonl(document_info_path, [])

    output_path = tmp_path / "processed" / "pages.jsonl"
    with pytest.raises(FileNotFoundError) as exc_info:
        ingest.write_pages_jsonl(
            benchmark_path=benchmark_path,
            document_info_path=document_info_path,
            raw_dir=raw_dir,
            pdf_dir=pdf_dir,
            output_path=output_path,
            workers=1,
        )

    assert str(exc_info.value) == "Missing 1 referenced PDFs: MISSING_2022_10K"
    assert not output_path.exists()
