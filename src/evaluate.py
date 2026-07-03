from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Protocol


IDENTITY_FIELDS = (
    "financebench_id",
    "company",
    "doc_name",
    "question_type",
    "question_reasoning",
    "domain_question_num",
    "question",
)

RAGAS_METRIC_KEYS = (
    "context_precision",
    "context_recall",
    "response_relevancy",
    "faithfulness",
)


def _progress(message: str) -> None:
    print(f"[ragas-eval] {message}", file=sys.stderr, flush=True)


class RagasEvaluator(Protocol):
    def evaluate(
        self,
        samples: list[dict[str, Any]],
        *,
        metrics: tuple[str, ...],
    ) -> list[dict[str, float]]:
        ...


class DefaultRagasEvaluator:
    def __init__(
        self,
        *,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        client: Any | None = None,
        openai_timeout: float | None = None,
        openai_max_retries: int | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.client = client
        self.openai_timeout = openai_timeout
        self.openai_max_retries = openai_max_retries

    def evaluate(
        self,
        samples: list[dict[str, Any]],
        *,
        metrics: tuple[str, ...],
    ) -> list[dict[str, float]]:
        _progress(
            f"initializing Ragas evaluator with {len(samples)} sample(s) "
            f"and metrics: {', '.join(metrics)}"
        )
        metric_objects = _ragas_metric_objects(
            metrics,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            client=self.client,
            openai_timeout=self.openai_timeout,
            openai_max_retries=self.openai_max_retries,
        )
        rows = [_ragas_dataset_row(sample) for sample in samples]
        _progress(f"converted {len(rows)} sample(s) to Ragas input rows")
        return _score_collection_metrics(rows, metric_objects, metrics)


def _ragas_metric_objects(
    metrics: tuple[str, ...],
    *,
    llm_model: str,
    embedding_model: str,
    client: Any | None,
    openai_timeout: float | None,
    openai_max_retries: int | None,
) -> list[Any]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ragas evaluation requires ragas and openai packages."
        ) from exc

    if client is None:
        openai_kwargs: dict[str, float | int] = {}
        if openai_timeout is not None:
            openai_kwargs["timeout"] = openai_timeout
        if openai_max_retries is not None:
            openai_kwargs["max_retries"] = openai_max_retries
        _progress(f"creating AsyncOpenAI client with options: {openai_kwargs}")
        openai_client = AsyncOpenAI(**openai_kwargs)
    else:
        _progress("using injected OpenAI client")
        openai_client = client
    _progress(f"creating Ragas LLM wrapper: {llm_model}")
    llm = llm_factory(llm_model, client=openai_client)
    _progress(f"creating Ragas embedding wrapper: {embedding_model}")
    embeddings = embedding_factory(
        provider="openai",
        model=embedding_model,
        client=openai_client,
        interface="modern",
    )
    metric_factories = {
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
        "response_relevancy": lambda: AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        ),
        "faithfulness": lambda: Faithfulness(llm=llm),
    }

    metric_objects: list[Any] = []
    for metric_key in metrics:
        metric_factory = metric_factories.get(metric_key)
        if metric_factory is None:
            raise RuntimeError(f"Ragas metric is unavailable: {metric_key}")
        _progress(f"creating Ragas metric object: {metric_key}")
        metric_objects.append(metric_factory())
    return metric_objects


def _default_ragas_evaluator(args: argparse.Namespace) -> DefaultRagasEvaluator:
    evaluator_kwargs: dict[str, Any] = {}
    if args.ragas_llm_model is not None:
        evaluator_kwargs["llm_model"] = args.ragas_llm_model
    if args.ragas_embedding_model is not None:
        evaluator_kwargs["embedding_model"] = args.ragas_embedding_model
    if getattr(args, "openai_timeout", None) is not None:
        evaluator_kwargs["openai_timeout"] = args.openai_timeout
    if getattr(args, "openai_max_retries", None) is not None:
        evaluator_kwargs["openai_max_retries"] = args.openai_max_retries
    return DefaultRagasEvaluator(**evaluator_kwargs)


def _score_collection_metrics(
    rows: list[dict[str, Any]],
    metric_objects: list[Any],
    metrics: tuple[str, ...],
) -> list[dict[str, float]]:
    if len(metric_objects) != len(metrics):
        raise RuntimeError("Ragas metric count did not match requested metrics.")

    score_rows = [dict[str, float]() for _ in rows]

    _progress(
        f"starting Ragas scoring for {len(rows)} row(s) "
        f"and {len(metrics)} metric(s)"
    )
    for metric_index, (metric_key, metric_object) in enumerate(
        zip(metrics, metric_objects),
        start=1,
    ):
        _progress(f"metric {metric_index}/{len(metrics)} started: {metric_key}")
        metric_inputs = [
            _collection_metric_input(metric_key, row)
            for row in rows
        ]
        metric_results = _score_collection_metric(
            metric_object,
            metric_inputs,
            metric_key,
        )
        if len(metric_results) != len(rows):
            raise RuntimeError(
                f"Ragas metric returned {len(metric_results)} scores for "
                f"{len(rows)} input rows: {metric_key}"
            )
        for score_row, metric_result in zip(score_rows, metric_results):
            score_row[metric_key] = _metric_result_value(metric_result)
        _progress(f"metric {metric_index}/{len(metrics)} finished: {metric_key}")

    return score_rows


def _score_collection_metric(
    metric_object: Any,
    metric_inputs: list[dict[str, Any]],
    metric_key: str,
) -> list[Any]:
    score = getattr(metric_object, "score", None)
    if callable(score):
        _progress(
            f"{metric_key}: using score() sequentially for "
            f"{len(metric_inputs)} row(s)"
        )
        metric_results = []
        for row_index, metric_input in enumerate(metric_inputs, start=1):
            _progress(
                f"{metric_key}: scoring row "
                f"{row_index}/{len(metric_inputs)}"
            )
            metric_results.append(score(**metric_input))
        return metric_results

    batch_score = getattr(metric_object, "batch_score", None)
    if callable(batch_score):
        _progress(
            f"{metric_key}: using batch_score() for {len(metric_inputs)} row(s)"
        )
        return list(batch_score(metric_inputs))

    abatch_score = getattr(metric_object, "abatch_score", None)
    if callable(abatch_score):
        _progress(
            f"{metric_key}: using abatch_score() for {len(metric_inputs)} row(s)"
        )
        return list(asyncio.run(abatch_score(metric_inputs)))

    ascore = getattr(metric_object, "ascore", None)
    if callable(ascore):
        _progress(
            f"{metric_key}: using ascore() sequentially for "
            f"{len(metric_inputs)} row(s)"
        )
        metric_results = []
        for row_index, metric_input in enumerate(metric_inputs, start=1):
            _progress(
                f"{metric_key}: async scoring row "
                f"{row_index}/{len(metric_inputs)}"
            )
            metric_results.append(asyncio.run(ascore(**metric_input)))
        return metric_results

    raise RuntimeError(
        f"Ragas collection metric does not support direct scoring: {metric_key}"
    )


def _collection_metric_input(
    metric_key: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    metric_fields = {
        "context_precision": ("user_input", "retrieved_contexts", "reference"),
        "context_recall": ("user_input", "retrieved_contexts", "reference"),
        "response_relevancy": ("user_input", "response"),
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
    }
    fields = metric_fields.get(metric_key)
    if fields is None:
        raise RuntimeError(f"Ragas metric is unavailable: {metric_key}")
    return {field: row[field] for field in fields}


def _metric_result_value(metric_result: Any) -> float:
    value = getattr(metric_result, "value", metric_result)
    return float(value)


def _ragas_dataset_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_input": sample["question"],
        "response": sample["answer"],
        "retrieved_contexts": sample["contexts"],
        "reference": sample["ground_truth"],
    }


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


def _top_chunks(record: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    chunks = record.get("retrieved_chunks") or []
    if not isinstance(chunks, list):
        return []
    return [chunk for chunk in chunks[:top_k] if isinstance(chunk, dict)]


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _identity_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in IDENTITY_FIELDS
        if field in record
    }


def _ragas_sample(record: dict[str, Any], top_k: int) -> dict[str, Any]:
    ground_truth = _text_value(record.get("answer"))
    justification = _text_value(record.get("justification"))
    if justification:
        ground_truth = f"Answer: {ground_truth}\nJustification: {justification}"

    return {
        "question": _text_value(record.get("question")),
        "answer": _text_value(record.get("generated_answer")),
        "contexts": [
            _text_value(chunk.get("text"))
            for chunk in _top_chunks(record, top_k)
        ],
        "ground_truth": ground_truth,
    }


def _ragas_metrics_records(
    records: list[dict[str, Any]],
    top_k: int,
    evaluator: RagasEvaluator,
) -> list[dict[str, Any]]:
    samples = [_ragas_sample(record, top_k) for record in records]
    score_rows = evaluator.evaluate(samples, metrics=RAGAS_METRIC_KEYS)
    if len(score_rows) != len(records):
        raise ValueError(
            "Ragas evaluator returned a different number of score rows than inputs."
        )

    output_records: list[dict[str, Any]] = []
    for record, score_row in zip(records, score_rows):
        output_record = _identity_record(record)
        output_record.update(score_row)
        output_records.append(output_record)
    return output_records


def write_metrics_jsonl(
    input_path: Path,
    output_path: Path,
    top_k: int,
    evaluator: RagasEvaluator | None = None,
) -> int:
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")

    _progress(f"reading input JSONL: {input_path}")
    records = _read_jsonl(input_path)
    _progress(f"loaded {len(records)} input record(s); top_k={top_k}")
    ragas_evaluator = evaluator if evaluator is not None else DefaultRagasEvaluator()
    _progress("building Ragas metrics records")
    output_records = _ragas_metrics_records(records, top_k, ragas_evaluator)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")

    try:
        _progress(f"writing metrics JSONL: {output_path}")
        with tmp_output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for output_record in output_records:
                output_file.write(
                    json.dumps(output_record, ensure_ascii=False)
                    + "\n"
                )
        tmp_output_path.replace(output_path)
        _progress(f"finished writing {len(output_records)} output record(s)")
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    return len(records)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate FinanceBench answer JSONL rows and write metrics JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "answers" / "answers.jsonl",
        help="Input answer JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "eval" / "metrics.jsonl",
        help="Output metrics JSONL path.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--ragas-llm-model",
        default=None,
        help="OpenAI model used by Ragas collection metrics.",
    )
    parser.add_argument(
        "--ragas-embedding-model",
        default=None,
        help="OpenAI embedding model used by Ragas AnswerRelevancy.",
    )
    parser.add_argument(
        "--openai-timeout",
        type=float,
        default=None,
        help="Timeout passed to the default OpenAI client.",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=None,
        help="Max retries passed to the default OpenAI client.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written_count = write_metrics_jsonl(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        evaluator=_default_ragas_evaluator(args),
    )
    print(f"Wrote metrics for {written_count} answers to {args.output}")


if __name__ == "__main__":
    main()
