# FinanceBench RAG Evaluation

Python pipeline for evaluating retrieval-augmented generation (RAG) on FinanceBench financial question answering. The project ingests SEC filing PDFs, builds a Chroma vector index, retrieves evidence for financial questions, generates cited answers, and evaluates answer quality with Ragas metrics.

## What This Project Does

- Extracts page-level text and metadata from FinanceBench-referenced PDFs.
- Supports multiple chunking strategies, including page, paragraph, sentence, word, token, recursive, and character chunking.
- Embeds chunks with OpenAI embeddings and persists them in Chroma.
- Retrieves top-k evidence chunks with optional document-name filters or inferred company/year metadata filters.
- Generates JSON answers with citations back to retrieved document pages.
- Evaluates output with Ragas context precision, context recall, response relevancy, and faithfulness.
- Saves run-specific artifacts so chunking, retrieval, answering, and evaluation outputs can be inspected after each experiment.

## Current Saved Run

The main saved run is stored at:

```text
data/runs/page-128-overlap-100-top-10-embed-text-embedding-3-small-answer-gpt-5.4-nano-infer-filter/
```

Run configuration:

| Setting | Value |
| --- | --- |
| Questions evaluated | 150 |
| PDFs available locally | 368 |
| Unique documents extracted | 84 |
| Extracted page records | 12,013 |
| Chunking strategy | Page-level chunks |
| Retrieval backend | Chroma |
| Retrieval top-k | 10 |
| Embedding model | text-embedding-3-small |
| Answer model | gpt-5.4-nano |
| Ragas evaluator LLM | gpt-4o-mini |
| Ragas evaluator embeddings | text-embedding-3-small |

Aggregate results from `metrics_summary.csv`:

| Metric | Value |
| --- | ---: |
| Average context precision | 0.570 |
| Average context recall | 0.554 |
| Average response relevancy | 0.604 |
| Average faithfulness | 0.657 |
| Target document in top-k | 95.3% |
| Exact gold evidence page in top-k | 60.7% |
| Table or numeric-heavy questions | 86.7% |

The summary highlights a realistic bottleneck for financial RAG: target documents are usually retrieved, but exact evidence-page recall is still difficult, especially for table-heavy and numeric reasoning questions.

## Project Structure

```text
src/
  ingest.py        Extract referenced PDFs into page-level JSONL.
  chunk.py         Convert page records into chunk records.
  embed_chunks.py Embed chunks and persist vectors in Chroma.
  retrieve.py     Retrieve top-k chunks for FinanceBench questions.
  answer.py       Generate cited answers from retrieved chunks.
  evaluate.py     Score generated answers with Ragas metrics.
  pipeline.py     Orchestrate chunk -> embed -> retrieve -> answer -> evaluate.

tests/
  test_ingest.py
  test_chunk.py
  test_retrieve.py
  test_answer.py
  test_evaluate.py
  test_pipeline.py

data/
  raw/pdfs/        Local PDF corpus.
  processed/       Extracted page JSONL.
  runs/            Run-specific artifacts and metrics.

eval-set/
  financebench_open_source.jsonl
  financebench_document_information.jsonl
```

## Setup

Use the project conda environment:

```powershell
conda activate rag-eval-env
```

Set an OpenAI API key before running stages that call OpenAI:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

This repository does not currently include a dependency lockfile. The code imports `pypdf`, `chromadb`, `openai`, `ragas`, `langchain-text-splitters`, `tiktoken`, and `pytest`.

## Run The Pipeline

Ingest referenced PDFs into page-level JSONL:

```powershell
conda run -n rag-eval-env python -m src.ingest
```

Run the end-to-end workflow with the saved-run parameters:

```powershell
conda run -n rag-eval-env python -m src.pipeline --strategy page --chunk-size 128 --chunk-overlap 100 --top-k 10 --answer-model gpt-5.4-nano --infer-metadata-filter
```

Preview the commands without executing API-backed stages:

```powershell
conda run -n rag-eval-env python -m src.pipeline --strategy page --chunk-size 128 --chunk-overlap 100 --top-k 10 --answer-model gpt-5.4-nano --infer-metadata-filter --dry-run
```

## Test

```powershell
conda run -n rag-eval-env python -m pytest
```

The tests focus on core behavior: JSONL validation, PDF page extraction task building, deterministic chunk IDs, chunking strategies, embedding retry logic, Chroma retrieval filters, answer validation, pipeline planning, and Ragas evaluator wiring.

## Resume Bullets

- Built an end-to-end FinanceBench RAG evaluation pipeline in Python, extracting 84 FinanceBench-referenced filings from a 368-PDF local corpus into 12,013 page-level records, indexing them in Chroma with OpenAI embeddings, and generating cited answers for 150 benchmark questions.
- Improved RAG quality by tuning chunking strategy, chunk size, overlap, top-k retrieval, metadata filtering, embedding model, and answer model settings, then comparing runs with Ragas metrics and retrieval hit-rate analysis.
- Implemented metadata-aware retrieval, deterministic chunk IDs, retry-safe embedding ingestion, structured cited answer generation, and focused pytest coverage for ingestion, chunking, retrieval, evaluation, and pipeline orchestration.

## Notes

- Full pipeline runs require OpenAI API access and may incur API cost.
- Current metrics show strong target-document retrieval but lower exact evidence-page recall, which is expected for table-heavy financial filings and is a clear area for future retrieval improvement.
- Some saved metric metadata is inferred from run directory names because the raw metrics JSONL does not store every run override.
