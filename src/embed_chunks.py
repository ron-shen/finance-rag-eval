from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
from openai import OpenAI

from src.ingest import ingest_chunks_to_chroma


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Embed chunk JSONL with OpenAI and persist vectors in Chroma."
    )
    parser.add_argument(
        "--chunks",
        dest="chunks_path",
        type=Path,
        default=project_root / "data" / "chunk" / "token-chunk.jsonl",
        help="Input chunk-level JSONL path.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=project_root / "data" / "vector" / "chroma",
        help="Directory where Chroma persists the vector database.",
    )
    parser.add_argument(
        "--collection",
        default="financebench_chunks",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks to embed per OpenAI request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Maximum number of retries for each rate-limited embedding batch.",
    )
    parser.add_argument(
        "--retry-min-seconds",
        type=float,
        default=1.0,
        help="Initial retry delay when OpenAI does not provide a retry hint.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=60.0,
        help="Maximum retry delay for rate-limited embedding batches.",
    )
    return parser.parse_args()


def build_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required to embed chunks."
        )
    return OpenAI(api_key=api_key)


def main() -> None:
    args = parse_args()
    args.persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(args.persist_dir))
    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"embedding_model": args.model},
    )
    inserted_count = ingest_chunks_to_chroma(
        chunks_path=args.chunks_path,
        embedding_client=build_openai_client(),
        collection=collection,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_min_seconds=args.retry_min_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
    print(
        f"Inserted {inserted_count} chunks into Chroma collection "
        f"{args.collection!r} at {args.persist_dir}"
    )


if __name__ == "__main__":
    main()
