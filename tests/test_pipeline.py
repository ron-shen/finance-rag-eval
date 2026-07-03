from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _stages_by_name(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stage["name"]: stage for stage in plan["stages"]}


def _command_arg(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_build_pipeline_plan_routes_outputs_through_run_directory(
    tmp_path: Path,
) -> None:
    from src import pipeline

    plan = pipeline.build_pipeline_plan(
        project_root=tmp_path,
        strategy="token",
        chunk_size=512,
        chunk_overlap=64,
        top_k=7,
        run_name="token-512-overlap-64-top-7",
    )

    run_dir = tmp_path / "data" / "runs" / "token-512-overlap-64-top-7"
    pages_path = tmp_path / "data" / "processed" / "pages.jsonl"
    chunks_path = run_dir / "chunks.jsonl"
    vector_dir = run_dir / "chroma"
    retrievals_path = run_dir / "retrievals.jsonl"
    answers_path = run_dir / "answers.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    questions_path = tmp_path / "eval-set" / "financebench_open_source.jsonl"

    assert plan["run_id"] == "token-512-overlap-64-top-7"
    assert plan["run_dir"] == run_dir
    assert plan["paths"] == {
        "pages": pages_path,
        "chunks": chunks_path,
        "vector_store": vector_dir,
        "questions": questions_path,
        "retrievals": retrievals_path,
        "answers": answers_path,
        "metrics": metrics_path,
    }

    assert [stage["name"] for stage in plan["stages"]] == [
        "chunk",
        "embed_chunks",
        "retrieve",
        "answer",
        "evaluate",
    ]

    stages = _stages_by_name(plan)
    chunk_command = stages["chunk"]["command"]
    assert chunk_command[:3] == [sys.executable, "-m", "src.chunk"]
    assert _command_arg(chunk_command, "--input") == str(pages_path)
    assert _command_arg(chunk_command, "--output") == str(chunks_path)
    assert _command_arg(chunk_command, "--strategy") == "token"
    assert _command_arg(chunk_command, "--chunk-size") == "512"
    assert _command_arg(chunk_command, "--chunk-overlap") == "64"

    embed_command = stages["embed_chunks"]["command"]
    assert embed_command[:3] == [sys.executable, "-m", "src.embed_chunks"]
    assert _command_arg(embed_command, "--chunks") == str(chunks_path)
    assert _command_arg(embed_command, "--persist-dir") == str(vector_dir)

    retrieve_command = stages["retrieve"]["command"]
    assert retrieve_command[:3] == [sys.executable, "-m", "src.retrieve"]
    assert _command_arg(retrieve_command, "--questions") == str(questions_path)
    assert _command_arg(retrieve_command, "--persist-dir") == str(vector_dir)
    assert _command_arg(retrieve_command, "--output") == str(retrievals_path)
    assert _command_arg(retrieve_command, "--top-k") == "7"

    answer_command = stages["answer"]["command"]
    assert answer_command[:3] == [sys.executable, "-m", "src.answer"]
    assert _command_arg(answer_command, "--retrievals") == str(retrievals_path)
    assert _command_arg(answer_command, "--output") == str(answers_path)

    evaluate_command = stages["evaluate"]["command"]
    assert evaluate_command[:3] == [sys.executable, "-m", "src.evaluate"]
    assert _command_arg(evaluate_command, "--input") == str(answers_path)
    assert _command_arg(evaluate_command, "--output") == str(metrics_path)
    assert _command_arg(evaluate_command, "--top-k") == "7"


def test_build_pipeline_plan_derives_stable_parameter_sensitive_run_id(
    tmp_path: Path,
) -> None:
    from src import pipeline

    first = pipeline.build_pipeline_plan(
        project_root=tmp_path,
        strategy="paragraph",
        chunk_size=400,
        chunk_overlap=50,
        top_k=5,
    )
    same_parameters = pipeline.build_pipeline_plan(
        project_root=tmp_path,
        strategy="paragraph",
        chunk_size=400,
        chunk_overlap=50,
        top_k=5,
    )
    different_chunking = pipeline.build_pipeline_plan(
        project_root=tmp_path,
        strategy="paragraph",
        chunk_size=500,
        chunk_overlap=50,
        top_k=5,
    )
    different_retrieval = pipeline.build_pipeline_plan(
        project_root=tmp_path,
        strategy="paragraph",
        chunk_size=400,
        chunk_overlap=50,
        top_k=10,
    )

    assert first["run_id"] == same_parameters["run_id"]
    assert first["run_dir"] == tmp_path / "data" / "runs" / first["run_id"]
    assert different_chunking["run_id"] != first["run_id"]
    assert different_retrieval["run_id"] != first["run_id"]


def test_build_pipeline_plan_can_enable_inferred_metadata_filter(
    tmp_path: Path,
) -> None:
    from src import pipeline

    plan = pipeline.build_pipeline_plan(
        project_root=tmp_path,
        strategy="token",
        chunk_size=128,
        chunk_overlap=32,
        top_k=5,
        infer_metadata_filter=True,
    )

    retrieve_command = _stages_by_name(plan)["retrieve"]["command"]

    assert "--infer-metadata-filter" in retrieve_command
    assert _command_arg(retrieve_command, "--metadata-filter-chunks") == str(
        plan["paths"]["chunks"]
    )
    assert plan["run_id"].endswith("-infer-filter")
