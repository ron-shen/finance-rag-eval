from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class AnswerGenerator(Protocol):
    def __call__(self, retrieval: dict[str, Any], *, model: str) -> dict[str, Any]:
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


def _validate_generation(generation: dict[str, Any]) -> None:
    if not isinstance(generation.get("answer"), str):
        raise ValueError("Generation must include an answer string.")

    citations = generation.get("citations")
    if not isinstance(citations, list) or not citations:
        raise ValueError(
            "Generation citations must be a non-empty list with doc_name and page."
        )

    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError(
                "Generation citations must be a non-empty list with doc_name and page."
            )
        doc_name = citation.get("doc_name")
        page = citation.get("page")
        if (
            not isinstance(doc_name, str)
            or not doc_name.strip()
            or page is None
            or page == ""
        ):
            raise ValueError(
                "Generation citations must be a non-empty list with doc_name and page."
            )


def write_answers_jsonl(
    *,
    retrievals_path: Path,
    output_path: Path,
    generator: AnswerGenerator,
    model: str,
    progress_fn: Callable[[str], None] | None = None,
) -> int:
    retrievals = _read_jsonl(retrievals_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")

    try:
        with tmp_output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for index, retrieval in enumerate(retrievals, start=1):
                if progress_fn is not None:
                    question_id = retrieval.get("financebench_id") or "unknown_id"
                    progress_fn(
                        f"[{index}/{len(retrievals)}] processing {question_id}: "
                        f"{retrieval.get('question')}"
                    )
                output_record = dict(retrieval)
                answer_start = time.perf_counter()
                generation = generator(retrieval, model=model)
                answer_latency_ms = (time.perf_counter() - answer_start) * 1000
                _validate_generation(generation)

                output_record["generated_answer"] = generation["answer"]
                output_record["citations"] = generation["citations"]
                output_record["answer_model"] = model
                output_record["answer_latency_ms"] = answer_latency_ms
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    return len(retrievals)


def _retrieval_prompt(retrieval: dict[str, Any]) -> str:
    chunks = retrieval.get("retrieved_chunks") or []
    chunk_lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_lines.append(
            "\n".join(
                [
                    f"[{index}] doc_name: {chunk.get('doc_name')}",
                    f"[{index}] page: {chunk.get('page')}",
                    f"[{index}] text: {chunk.get('text')}",
                ]
            )
        )

    return "\n\n".join(
        [
            f"Question: {retrieval.get('question')}",
            "Retrieved chunks:",
            "\n\n".join(chunk_lines),
        ]
    )


def build_openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required to generate answers."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def generate_openai_answer(
    retrieval: dict[str, Any],
    *,
    model: str,
    client: Any | None = None,
) -> dict[str, Any]:
    if client is None:
        client = build_openai_client()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer FinanceBench questions using only the retrieved chunks. "
                    "Return only a JSON object in this exact format:\n"
                    "{\n"
                    '  "answer": "3M\'s FY2018 capital expenditures were $1.577 billion.",\n'
                    '  "citations": [\n'
                    '    {"doc_name": "3M_2018_10K", "page": 59}\n'
                    "  ]\n"
                    "}\n"
                    "Citations must be objects containing doc_name and page from the supporting chunks.\n"
                    "If the answer is explicitly supported by the retrieved chunks, answer directly and cite the supporting chunks.\n"
                    "The citations array is required and must never be empty. Every answer, including 0, not found, not disclosed, or insufficient evidence, must cite at least one retrieved chunk that was reviewed.\n"
                    "If no retrieved chunk supports the requested answer, say the retrieved evidence is insufficient and cite the most relevant retrieved chunk(s) reviewed. Do not cite those chunks as support for the answer; cite them as reviewed evidence."    
                ),
            },
            {"role": "user", "content": _retrieval_prompt(retrieval)},
        ]
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI response did not include content.")
    generation = json.loads(content)
    if not isinstance(generation, dict):
        raise ValueError("OpenAI response must be a JSON object.")
    return generation


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate FinanceBench answers with citations from retrieval JSONL."
    )
    parser.add_argument(
        "--retrievals",
        type=Path,
        default=project_root / "data" / "retrieval" / "retrievals.jsonl",
        help="Input retrieval JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "answers" / "answers.jsonl",
        help="Output answer JSONL path.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="OpenAI model used for answer generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_count = write_answers_jsonl(
        retrievals_path=args.retrievals,
        output_path=args.output,
        generator=generate_openai_answer,
        model=args.model,
        progress_fn=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(f"Wrote answers for {written_count} retrievals to {args.output}")


if __name__ == "__main__":
    main()
