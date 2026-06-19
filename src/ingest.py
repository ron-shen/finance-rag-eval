from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
import sys
from pathlib import Path

from pypdf import PdfReader


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
