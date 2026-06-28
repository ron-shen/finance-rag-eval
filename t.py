from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from openai import OpenAI
import os

sys.stdout.reconfigure(encoding="utf-8")

CHROMA_PATH = Path("data/vector/chroma")
COLLECTION_NAME = "financebench_chunks"
EMBED_MODEL = "text-embedding-3-small"


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# print("Collections:")
# for collection in client.list_collections():
#     print(f"- {collection.name}: {collection.count()} rows, metadata={collection.metadata}")

collection = client.get_collection(COLLECTION_NAME)

# print("\nCount:")
# print(collection.count())

# print("\nSample rows:")
# sample = collection.get(
#     limit=1,
#     include=["documents", "metadatas", "embeddings"],
# )
# print_json(sample)

# print("\nFiltered rows for one document:")
# filtered = collection.get(
#     where={"doc_name": "3M_2018_10K"},
#     limit=5,
#     include=["documents", "metadatas"],
# )
# print_json(filtered)

print("\nSemantic query:")
query = "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
query_embedding = OpenAI().embeddings.create(
    model=EMBED_MODEL,
    input=[query],
).data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
print_json(results)