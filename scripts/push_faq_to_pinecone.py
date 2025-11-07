"""Utility script to ingest FAQ markdown into Pinecone."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


def _read_chunks(path: Path, *, separator: str = "\n\n") -> List[str]:
    data = path.read_text(encoding="utf-8")
    parts = [segment.strip() for segment in data.split(separator)]
    return [segment for segment in parts if segment]


def _batch(iterable: Iterable, size: int) -> Iterable[List]:
    chunk: List = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Push FAQ entries into a Pinecone index")
    parser.add_argument("faq_path", nargs="?", default=os.getenv("FAQ_SOURCE_PATH", "kb/urbanstyle_faq.md"))
    parser.add_argument("--index", dest="index_name", default=os.getenv("PINECONE_INDEX"))
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("PINECONE_API_KEY"))
    parser.add_argument("--environment", dest="environment", default=os.getenv("PINECONE_ENV"))
    parser.add_argument("--batch-size", type=int, default=64, help="Number of FAQ entries per upsert batch")
    parser.add_argument("--embedding-model", default=os.getenv("PINECONE_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"))
    args = parser.parse_args()

    if not args.api_key or not args.index_name or not args.environment:
        raise SystemExit("Set PINECONE_API_KEY, PINECONE_INDEX, dan PINECONE_ENV sebelum menjalankan skrip")

    faq_path = Path(args.faq_path)
    if not faq_path.exists():
        raise SystemExit(f"FAQ source file tidak ditemukan: {faq_path}")

    print(f"Memuat FAQ dari {faq_path} ...")
    chunks = _read_chunks(faq_path)
    if not chunks:
        raise SystemExit("Tidak ada konten FAQ yang ditemukan")

    client = Pinecone(api_key=args.api_key, environment=args.environment)
    index = client.Index(args.index_name)
    embedder = SentenceTransformer(args.embedding_model)

    total = 0
    for chunk_batch in _batch(chunks, args.batch_size):
        vectors = []
        embeddings = embedder.encode(chunk_batch, convert_to_numpy=False, normalize_embeddings=True)
        for text, embedding in zip(chunk_batch, embeddings):
            vectors.append(
                {
                    "id": str(uuid4()),
                    "values": list(map(float, embedding)),
                    "metadata": {"text": text},
                }
            )
        upsert_response = index.upsert(vectors=vectors)
        total += len(vectors)
        print(f"Upsert {len(vectors)} records -> {upsert_response}")

    print(f"Selesai. Total FAQ yang diunggah: {total}")


if __name__ == "__main__":
    main()
