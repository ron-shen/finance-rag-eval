from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)


SUPPORTED_STRATEGIES = {
    "page",
    "paragraph",
    "sentence",
    "word",
    "token",
    "recursive",
    "recursive_character",
    "character",
}


class TextSplitter(Protocol):
    def split_text(self, text: str) -> list[str]:
        ...


def _normalize_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(
            f"Unsupported chunking strategy: {strategy}. "
            f"Supported strategies: {supported}."
        )
    return normalized


def _validate_chunking_args(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            records.append(record)
    return records


def _clean_chunks(chunks: list[str]) -> list[str]:
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_words(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

    return chunks


def _split_with_regex_separator(
    text: str,
    separator_pattern: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    splits = _clean_chunks(re.split(separator_pattern, text))
    chunks: list[str] = []

    for split in splits:
        if len(split) <= chunk_size:
            chunks.append(split)
        else:
            chunks.extend(_split_oversized_text(split, chunk_size, chunk_overlap))

    return chunks


def _split_characters(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _split_oversized_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    splitter = _build_langchain_splitter("recursive", chunk_size, chunk_overlap)
    if splitter is not None:
        return _clean_chunks(splitter.split_text(text))
    return _split_characters(text, chunk_size, chunk_overlap)


def _fallback_split_text(
    text: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if strategy == "paragraph":
        return _split_with_regex_separator(text, r"\n\s*\n", chunk_size, chunk_overlap)
    if strategy == "sentence":
        return _split_with_regex_separator(
            text,
            r"(?<=[.!?])\s+",
            chunk_size,
            chunk_overlap,
        )
    return _split_characters(text, chunk_size, chunk_overlap)


def _build_langchain_splitter(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> TextSplitter | None:
    if strategy == "token":
        if TokenTextSplitter is None:
            raise ImportError(
                "The token strategy requires tiktoken. "
                "Install tiktoken to use strategy='token'."
            )
        try:
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except ImportError as exc:
            raise ImportError(
                "The token strategy requires tiktoken. "
                "Install tiktoken to use strategy='token'."
            ) from exc
    if (
        strategy in {"recursive", "recursive_character"}
        and RecursiveCharacterTextSplitter is not None
    ):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if strategy == "character" and CharacterTextSplitter is not None:
        return CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return None


def _split_text(
    text: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if strategy == "page":
        return [text]
    if strategy == "word":
        return _split_words(text, chunk_size, chunk_overlap)
    if strategy == "paragraph":
        return _split_with_regex_separator(text, r"\n\s*\n", chunk_size, chunk_overlap)
    if strategy == "sentence":
        return _split_with_regex_separator(
            text,
            r"(?<=[.!?])\s+",
            chunk_size,
            chunk_overlap,
        )

    # For token, recursive, recursive_character, and character strategies, use langchain splitters
    splitter = _build_langchain_splitter(strategy, chunk_size, chunk_overlap)
    if splitter is not None:
        return _clean_chunks(splitter.split_text(text))

    return _fallback_split_text(text, strategy, chunk_size, chunk_overlap)


def _chunk_id(
    record: dict[str, Any],
    record_index: int,
    chunk_index: int,
    chunk_text: str,
    strategy: str,
) -> str:
    identity = {
        "record_index": record_index,
        "chunk_index": chunk_index,
        "chunking_strategy": strategy,
        "doc_name": record.get("doc_name"),
        "page": record.get("page"),
        "source_path": record.get("source_path"),
        "text": chunk_text,
    }
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_chunks_jsonl(
    input_path: Path,
    output_path: Path,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    normalized_strategy = _normalize_strategy(strategy)
    _validate_chunking_args(chunk_size, chunk_overlap)

    records = _read_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    chunk_count = 0

    try:
        with tmp_output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record_index, record in enumerate(records):
                text = record.get("text") or ""
                chunks = _split_text(
                    str(text),
                    normalized_strategy,
                    chunk_size,
                    chunk_overlap,
                )
                for chunk_index, chunk_text in enumerate(chunks):
                    chunk_record = dict(record)
                    chunk_record["text"] = chunk_text
                    chunk_record["chunk_index"] = chunk_index
                    chunk_record["chunk_id"] = _chunk_id(
                        record,
                        record_index,
                        chunk_index,
                        chunk_text,
                        normalized_strategy,
                    )
                    chunk_record["chunking_strategy"] = normalized_strategy
                    output_file.write(
                        json.dumps(chunk_record, ensure_ascii=False) + "\n"
                    )
                    chunk_count += 1
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    return chunk_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Chunk page-level JSONL into chunk-level JSONL."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=project_root / "data" / "processed" / "pages.jsonl",
        help="Input page-level JSONL path.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=project_root / "data" / "processed" / "chunks.jsonl",
        help="Output chunk-level JSONL path.",
    )
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=sorted(SUPPORTED_STRATEGIES | {"recursive-character"}),
        help="Chunking strategy to try.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help=(
            "Max characters per chunk, max words when --strategy word is used, "
            "or max tokens when --strategy token is used. "
            "Ignored for --strategy page."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help=(
            "Overlap size in characters, words when --strategy word is used, "
            "or tokens when --strategy token is used. "
            "Ignored for --strategy page."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_count = write_chunks_jsonl(
        input_path=args.input_path,
        output_path=args.output_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Wrote {chunk_count} chunks to {args.output_path}")


if __name__ == "__main__":
    main()
