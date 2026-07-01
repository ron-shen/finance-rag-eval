from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_COLLECTION = "financebench_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_ANSWER_MODEL = "gpt-5.4-nano"


def _run_id(
    *,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    ragas_llm_model: str | None = None,
    ragas_embedding_model: str | None = None,
    filter_doc_name: bool = False,
) -> str:
    normalized_strategy = strategy.strip().lower().replace("_", "-")
    parts = [
        f"{normalized_strategy}-{chunk_size}",
        f"overlap-{chunk_overlap}",
        f"top-{top_k}",
        f"embed-{_slug(embedding_model)}",
        f"answer-{_slug(answer_model)}",
    ]
    if ragas_llm_model is not None:
        parts.append(f"ragas-llm-{_slug(ragas_llm_model)}")
    if ragas_embedding_model is not None:
        parts.append(f"ragas-embed-{_slug(ragas_embedding_model)}")
    if filter_doc_name:
        parts.append("filter-doc")
    return "-".join(parts)


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {".", "-"} else "-"
        for character in value.strip().lower()
    ).strip("-")


def _add_option(command: list[str], option: str, value: Any | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def _add_flag(command: list[str], option: str, enabled: bool) -> None:
    if enabled:
        command.append(option)


def build_pipeline_plan(
    *,
    project_root: Path,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    run_name: str | None = None,
    pages_path: Path | None = None,
    questions_path: Path | None = None,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    ragas_llm_model: str | None = None,
    ragas_embedding_model: str | None = None,
    filter_doc_name: bool = False,
    openai_timeout: float | None = None,
    openai_max_retries: int | None = None,
) -> dict[str, Any]:
    run_id = run_name or _run_id(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embedding_model=embedding_model,
        answer_model=answer_model,
        ragas_llm_model=ragas_llm_model,
        ragas_embedding_model=ragas_embedding_model,
        filter_doc_name=filter_doc_name,
    )
    run_dir = project_root / "data" / "runs" / run_id
    paths = {
        "pages": pages_path or project_root / "data" / "processed" / "pages.jsonl",
        "chunks": run_dir / "chunks.jsonl",
        "vector_store": run_dir / "chroma",
        "questions": questions_path
        or project_root / "eval-set" / "financebench_open_source.jsonl",
        "retrievals": run_dir / "retrievals.jsonl",
        "answers": run_dir / "answers.jsonl",
        "metrics": run_dir / "metrics.jsonl",
    }

    embed_command = [
        sys.executable,
        "-m",
        "src.embed_chunks",
        "--chunks",
        str(paths["chunks"]),
        "--persist-dir",
        str(paths["vector_store"]),
        "--collection",
        collection,
        "--model",
        embedding_model,
    ]
    retrieve_command = [
        sys.executable,
        "-m",
        "src.retrieve",
        "--questions",
        str(paths["questions"]),
        "--persist-dir",
        str(paths["vector_store"]),
        "--collection",
        collection,
        "--model",
        embedding_model,
        "--output",
        str(paths["retrievals"]),
        "--top-k",
        str(top_k),
    ]
    _add_flag(retrieve_command, "--filter-doc-name", filter_doc_name)

    evaluate_command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--input",
        str(paths["answers"]),
        "--output",
        str(paths["metrics"]),
        "--top-k",
        str(top_k),
    ]
    _add_option(evaluate_command, "--ragas-llm-model", ragas_llm_model)
    _add_option(evaluate_command, "--ragas-embedding-model", ragas_embedding_model)
    _add_option(evaluate_command, "--openai-timeout", openai_timeout)
    _add_option(evaluate_command, "--openai-max-retries", openai_max_retries)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "project_root": project_root,
        "paths": paths,
        "stages": [
            {
                "name": "chunk",
                "command": [
                    sys.executable,
                    "-m",
                    "src.chunk",
                    "--input",
                    str(paths["pages"]),
                    "--output",
                    str(paths["chunks"]),
                    "--strategy",
                    strategy,
                    "--chunk-size",
                    str(chunk_size),
                    "--chunk-overlap",
                    str(chunk_overlap),
                ],
            },
            {
                "name": "embed_chunks",
                "command": embed_command,
            },
            {
                "name": "retrieve",
                "command": retrieve_command,
            },
            {
                "name": "answer",
                "command": [
                    sys.executable,
                    "-m",
                    "src.answer",
                    "--retrievals",
                    str(paths["retrievals"]),
                    "--output",
                    str(paths["answers"]),
                    "--model",
                    answer_model,
                ],
            },
            {
                "name": "evaluate",
                "command": evaluate_command,
            },
        ],
    }


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return " ".join(_quote_posix(argument) for argument in command)


def _quote_posix(argument: str) -> str:
    if argument and all(character.isalnum() or character in "-_./:" for character in argument):
        return argument
    return "'" + argument.replace("'", "'\"'\"'") + "'"


def run_pipeline(
    plan: dict[str, Any],
    *,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    progress_fn: Callable[[str], None] = print,
) -> None:
    if not dry_run:
        plan["run_dir"].mkdir(parents=True, exist_ok=True)
    for stage in plan["stages"]:
        command = stage["command"]
        progress_fn(f"[pipeline] {stage['name']}: {format_command(command)}")
        if not dry_run:
            runner(command, check=True, cwd=plan["project_root"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the FinanceBench RAG workflow after ingest: "
            "chunk -> embed_chunks -> retrieve -> answer -> evaluate."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--pages",
        type=Path,
        default=None,
        help="Page-level JSONL produced by ingest. Defaults to data/processed/pages.jsonl.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="FinanceBench question JSONL. Defaults to eval-set/financebench_open_source.jsonl.",
    )
    parser.add_argument(
        "--strategy",
        default="token",
        choices=[
            "page",
            "paragraph",
            "sentence",
            "word",
            "token",
            "recursive",
            "recursive_character",
            "recursive-character",
            "character",
        ],
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--ragas-llm-model", default=None)
    parser.add_argument("--ragas-embedding-model", default=None)
    parser.add_argument("--openai-timeout", type=float, default=None)
    parser.add_argument("--openai-max-retries", type=int, default=None)
    parser.add_argument(
        "--filter-doc-name",
        action="store_true",
        help="Restrict retrieval to chunks from each question's doc_name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stage commands without running them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_pipeline_plan(
        project_root=args.project_root,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        run_name=args.run_name,
        pages_path=args.pages,
        questions_path=args.questions,
        collection=args.collection,
        embedding_model=args.embedding_model,
        answer_model=args.answer_model,
        ragas_llm_model=args.ragas_llm_model,
        ragas_embedding_model=args.ragas_embedding_model,
        filter_doc_name=args.filter_doc_name,
        openai_timeout=args.openai_timeout,
        openai_max_retries=args.openai_max_retries,
    )
    print(f"[pipeline] run_id: {plan['run_id']}")
    print(f"[pipeline] run_dir: {plan['run_dir']}")
    run_pipeline(plan, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
