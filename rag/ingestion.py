from __future__ import annotations
import json, re
from pathlib import Path
from .config import RAGConfig
from .schemas import DocumentMetadata
from .chunking import recursive_chunks
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore

META_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

def parse_markdown(path: Path):
    text = path.read_text(encoding="utf-8")
    m = META_RE.match(text)
    meta = {}
    body = text
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = text[m.end():]
    meta.setdefault("document_id", path.stem)
    meta.setdefault("title", path.stem.replace("_", " ").title())
    meta.setdefault("source_path", str(path))
    if "version" in meta:
        meta["version"] = int(meta["version"])
    return DocumentMetadata.model_validate(meta), body

def ingest(config: RAGConfig | None = None):
    config = config or RAGConfig()
    docs = []
    for path in sorted(config.knowledge_base_dir.glob("*")):
        if path.suffix.lower() == ".md":
            docs.append(parse_markdown(path))
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data if isinstance(data, list) else [data]
            for x in entries:
                body = x.pop("text")
                docs.append((DocumentMetadata.model_validate(x), body))

    chunks = []
    for meta, body in docs:
        chunks.extend(recursive_chunks(body, meta, config.chunk_size, config.chunk_overlap))

    embedder = EmbeddingModel(config.embedding_model)
    vectors = embedder.encode([c.text for c in chunks])
    store = FAISSVectorStore()
    store.build(vectors, chunks)
    store.save(config.index_dir)
    print(f"Ingested {len(docs)} documents -> {len(chunks)} unique chunks")
    print(f"Saved FAISS index to {config.index_dir}")

if __name__ == "__main__":
    ingest()
