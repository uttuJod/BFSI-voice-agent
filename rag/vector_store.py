from __future__ import annotations
import json
from pathlib import Path
import faiss
import numpy as np
from .schemas import Chunk, DocumentMetadata, DocumentStatus, RetrievedChunk, SearchFilters

class FAISSVectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, vectors: np.ndarray, chunks: list[Chunk]):
        if len(chunks) != len(vectors):
            raise ValueError("Vector/chunk count mismatch")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in self.chunks], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path):
        obj = cls()
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        obj.chunks = [Chunk.model_validate(x) for x in raw]
        return obj

    def search(self, query_vector: np.ndarray, top_k: int, score_threshold: float | None = None,
               filters: SearchFilters | None = None):
        fetch_k = min(max(top_k * 5, top_k), len(self.chunks))
        scores, ids = self.index.search(query_vector.reshape(1, -1), fetch_k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            c = self.chunks[int(idx)]
            if score_threshold is not None and float(score) < score_threshold:
                continue
            if filters:
                if filters.domain and c.metadata.domain != filters.domain:
                    continue
                if filters.status and c.metadata.status != filters.status:
                    continue
                if filters.document_ids and c.document_id not in filters.document_ids:
                    continue
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=float(score),
                metadata=c.metadata,
                rank=len(out) + 1,
            ))
            if len(out) >= top_k:
                break
        return out
