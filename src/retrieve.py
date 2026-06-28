from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


Scorer = Callable[[dict[str, Any], dict[str, Any]], float]


class EmbeddingClient(Protocol):
    embeddings: Any


class ChromaCollection(Protocol):
    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


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


def _retrieved_chunk(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "doc_name": chunk["doc_name"],
        "page": chunk["page"],
        "score": score,
        "text": chunk["text"],
    }


def write_retrievals_jsonl(
    questions_path: Path,
    chunks: Sequence[dict[str, Any]],
    output_path: Path,
    scorer: Scorer,
    top_k: int,
) -> int:
    questions = _read_jsonl(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")

    try:
        with tmp_output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for question in questions:
                scored_chunks = [
                    (scorer(question, chunk), chunk)
                    for chunk in chunks
                ]
                ranked_chunks = sorted(
                    scored_chunks,
                    key=lambda scored_chunk: scored_chunk[0],
                    reverse=True,
                )
                output_record = dict(question)
                output_record["retrieved_chunks"] = [
                    _retrieved_chunk(chunk, score)
                    for score, chunk in ranked_chunks[:top_k]
                ]
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    return len(questions)


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def retrieve_chroma_top_k(
    question: dict[str, Any],
    *,
    collection: ChromaCollection,
    embedding_client: EmbeddingClient,
    model: str,
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query_text = str(question.get("question") or "")
    response = embedding_client.embeddings.create(model=model, input=[query_text])
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    retrieved: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        chunk = dict(metadata or {})
        chunk["text"] = document or ""
        retrieved.append(_retrieved_chunk(chunk, _distance_to_score(float(distance))))
    return retrieved


def write_chroma_retrievals_jsonl(
    *,
    questions_path: Path,
    output_path: Path,
    collection: ChromaCollection,
    embedding_client: EmbeddingClient,
    model: str,
    top_k: int,
    filter_doc_name: bool = False,
) -> int:
    questions = _read_jsonl(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")

    try:
        with tmp_output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for question in questions:
                where = (
                    {"doc_name": question["doc_name"]}
                    if filter_doc_name and question.get("doc_name")
                    else None
                )
                output_record = dict(question)
                output_record["retrieved_chunks"] = retrieve_chroma_top_k(
                    question,
                    collection=collection,
                    embedding_client=embedding_client,
                    model=model,
                    top_k=top_k,
                    where=where,
                )
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    return len(questions)


def build_openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required to retrieve chunks."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def build_chroma_collection(persist_dir: Path, collection_name: str) -> Any:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(collection_name)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Retrieve top-k FinanceBench chunks from Chroma and write JSONL."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=project_root / "eval-set" / "financebench_open_source.jsonl",
        help="Input FinanceBench question JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "retrieval" / "retrievals.jsonl",
        help="Output JSONL path for retrieved chunks.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=project_root / "data" / "vector" / "chroma",
        help="Directory containing the persisted Chroma database.",
    )
    parser.add_argument(
        "--collection",
        default="financebench_chunks",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model for questions.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--filter-doc-name",
        action="store_true",
        help="Restrict each query to chunks from that question's doc_name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_count = write_chroma_retrievals_jsonl(
        questions_path=args.questions,
        output_path=args.output,
        collection=build_chroma_collection(args.persist_dir, args.collection),
        embedding_client=build_openai_client(),
        model=args.model,
        top_k=args.top_k,
        filter_doc_name=args.filter_doc_name,
    )
    print(f"Wrote retrievals for {written_count} questions to {args.output}")


if __name__ == "__main__":
    main()
