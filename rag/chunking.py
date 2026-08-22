from hashlib import sha256
from .schemas import Chunk, DocumentMetadata

def recursive_chunks(text: str, metadata: DocumentMetadata, chunk_size: int, overlap: int):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    idx = 0
    seen = set()
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        digest = sha256(f"{metadata.document_id}|{piece}".encode()).hexdigest()[:20]
        if digest not in seen:
            seen.add(digest)
            chunks.append(Chunk(
                chunk_id=f"{metadata.document_id}:{digest}",
                document_id=metadata.document_id,
                text=piece,
                metadata=metadata,
                chunk_index=idx,
            ))
            idx += 1
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
