from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader


class EmbeddingClient(Protocol):
    embeddings: Any


class ChromaCollection(Protocol):
    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...


def _batch_records(records: list[dict], batch_size: int) -> list[list[dict]]:
    return [
        records[start : start + batch_size]
        for start in range(0, len(records), batch_size)
    ]


def _chroma_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"chunk_id", "text"} or value is None:
            continue
        if isinstance(value, str | int | float | bool):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return metadata


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    error_type = error.__class__.__name__
    message = str(error).lower()
    return (
        status_code == 429
        or code == "rate_limit_exceeded"
        or error_type == "RateLimitError"
        or "rate limit" in message
    )


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return float(retry_after_ms) / 1000
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass

    message = str(error)
    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*ms", message, re.I)
    if match:
        return float(match.group(1)) / 1000

    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*s", message, re.I)
    if match:
        return float(match.group(1))

    return None


def _embedding_response_with_retries(
    *,
    embedding_client: EmbeddingClient,
    model: str,
    documents: list[str],
    max_retries: int,
    retry_min_seconds: float,
    retry_max_seconds: float,
    sleep_fn: Callable[[float], None],
) -> Any:
    attempt = 0
    while True:
        try:
            return embedding_client.embeddings.create(model=model, input=documents)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise

            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = retry_min_seconds * (2**attempt)
            sleep_fn(min(max(delay, 0), retry_max_seconds))
            attempt += 1


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return records


def load_referenced_docs(benchmark_path: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for record in read_jsonl(benchmark_path):
        doc_name = record["doc_name"]
        if doc_name not in docs:
            docs[doc_name] = {"company": record.get("company")}
    return docs


def load_document_info(document_info_path: Path) -> dict[str, dict]:
    return {
        record["doc_name"]: record
        for record in read_jsonl(document_info_path)
        if record.get("doc_name")
    }


def infer_period(doc_name: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", doc_name)
    return int(match.group(0)) if match else None


def build_page_record(
    *,
    doc_name: str,
    company: str | None,
    period: int | str | None,
    page_number: int,
    text: str,
    source_path: str,
) -> dict:
    return {
        "doc_name": doc_name,
        "company": company,
        "period": period,
        "page": page_number,
        "text": text,
        "source_path": source_path,
        "doc_type": "pdf",
    }


def ingest_chunks_to_chroma(
    *,
    chunks_path: Path,
    embedding_client: EmbeddingClient,
    collection: ChromaCollection,
    model: str,
    batch_size: int = 100,
    max_retries: int = 8,
    retry_min_seconds: float = 1.0,
    retry_max_seconds: float = 60.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater.")
    if retry_min_seconds < 0:
        raise ValueError("retry_min_seconds must be zero or greater.")
    if retry_max_seconds < 0:
        raise ValueError("retry_max_seconds must be zero or greater.")

    records = read_jsonl(chunks_path)
    if not records:
        return 0

    inserted_count = 0
    for batch in _batch_records(records, batch_size):
        ids = [str(record["chunk_id"]) for record in batch]
        documents = [str(record.get("text") or "") for record in batch]
        metadatas = [_chroma_metadata(record) for record in batch]

        response = _embedding_response_with_retries(
            embedding_client=embedding_client,
            model=model,
            documents=documents,
            max_retries=max_retries,
            retry_min_seconds=retry_min_seconds,
            retry_max_seconds=retry_max_seconds,
            sleep_fn=sleep_fn,
        )
        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(documents):
            raise ValueError(
                "Embedding response count does not match the number of input chunks."
            )
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        inserted_count += len(batch)

    return inserted_count


def extract_pdf_pages(task: dict) -> tuple[str, list[str], int]:
    doc_name = task["doc_name"]
    pdf_path = Path(task["pdf_path"])
    reader = PdfReader(str(pdf_path))
    lines = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        record = build_page_record(
            doc_name=doc_name,
            company=task["company"],
            period=task["period"],
            page_number=page_number,
            text=text,
            source_path=task["source_path"],
        )
        lines.append(json.dumps(record, ensure_ascii=False))

    return doc_name, lines, len(lines)


def write_pages_jsonl(
    *,
    benchmark_path: Path,
    document_info_path: Path,
    raw_dir: Path,
    pdf_dir: Path,
    output_path: Path,
    workers: int,
) -> tuple[int, int]:
    referenced_docs = load_referenced_docs(benchmark_path)
    document_info = load_document_info(document_info_path)

    missing_pdfs = [
        doc_name
        for doc_name in referenced_docs
        if not (pdf_dir / f"{doc_name}.pdf").exists()
    ]
    if missing_pdfs:
        missing = ", ".join(missing_pdfs[:10])
        suffix = "..." if len(missing_pdfs) > 10 else ""
        raise FileNotFoundError(f"Missing {len(missing_pdfs)} referenced PDFs: {missing}{suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    for doc_name in sorted(referenced_docs):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        metadata = document_info.get(doc_name, {})
        tasks.append(
            {
                "doc_name": doc_name,
                "pdf_path": str(pdf_path),
                "company": metadata.get("company")
                or referenced_docs[doc_name].get("company"),
                "period": metadata.get("doc_period") or infer_period(doc_name),
                "source_path": pdf_path.relative_to(raw_dir).as_posix(),
            }
        )

    results_by_doc: dict[str, list[str]] = {}
    page_count = 0

    if workers == 1:
        for doc_index, task in enumerate(tasks, start=1):
            print(
                f"[{doc_index}/{len(tasks)}] extracting {task['source_path']}",
                file=sys.stderr,
            )
            doc_name, lines, doc_pages = extract_pdf_pages(task)
            results_by_doc[doc_name] = lines
            page_count += doc_pages
    else:
        finished = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_pdf_pages, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                doc_name, lines, doc_pages = future.result()
                results_by_doc[doc_name] = lines
                page_count += doc_pages
                finished += 1
                print(
                    f"[{finished}/{len(tasks)}] extracted {task['source_path']} "
                    f"({doc_pages} pages)",
                    file=sys.stderr,
                )

    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for task in tasks:
            lines = results_by_doc[task["doc_name"]]
            output_file.write("\n".join(lines) + "\n")
    tmp_output_path.replace(output_path)

    return len(referenced_docs), page_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"

    parser = argparse.ArgumentParser(
        description="Extract referenced FinanceBench PDFs into page-level JSONL."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=project_root / "eval-set" / "financebench_open_source.jsonl",
    )
    parser.add_argument(
        "--document-info",
        type=Path,
        default=project_root / "eval-set" / "financebench_document_information.jsonl",
    )
    parser.add_argument("--raw-dir", type=Path, default=raw_dir)
    parser.add_argument("--pdf-dir", type=Path, default=raw_dir / "pdfs")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed" / "pages.jsonl",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of PDFs to extract in parallel. Use 1 for serial extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doc_count, page_count = write_pages_jsonl(
        benchmark_path=args.benchmark,
        document_info_path=args.document_info,
        raw_dir=args.raw_dir,
        pdf_dir=args.pdf_dir,
        output_path=args.output,
        workers=max(1, args.workers),
    )
    print(f"Wrote {page_count} pages from {doc_count} PDFs to {args.output}")


if __name__ == "__main__":
    main()
