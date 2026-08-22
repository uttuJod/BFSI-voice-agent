import os
from pathlib import Path
from pydantic import BaseModel, Field


def active_domain() -> str:
    """
    Domain pack selected by the DOMAIN environment variable.

    Each pack lives under domains/<name>/ and contains the knowledge base,
    evaluation datasets and (where applicable) seed business data. The
    router, guards and voice runtime are domain-agnostic; only data and
    evaluation change.
    """
    name = os.getenv("DOMAIN", "bfsi").strip().lower()
    if not (Path("domains") / name).is_dir():
        raise ValueError(
            f"Unknown DOMAIN={name!r}. Available: "
            + ", ".join(sorted(p.name for p in Path("domains").iterdir() if p.is_dir()))
        )
    return name


def domain_path(*parts: str) -> Path:
    return Path("domains").joinpath(active_domain(), *parts)


class RAGConfig(BaseModel):
    knowledge_base_dir: Path = Field(
        default_factory=lambda: domain_path("knowledge_base")
    )
    index_dir: Path = Field(
        default_factory=lambda: Path("results") / "faiss_index" / active_domain()
    )

    # Multilingual model so Hindi/Hinglish evaluation is meaningful.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=5, ge=1)

    # Retrieval and evidence evaluation are deliberately separate.
    # Retrieval keeps a wider candidate set; the evaluator decides safety.
    score_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    max_retrieval_iterations: int = Field(default=3, ge=1, le=10)

    low_relevance_threshold: float = Field(default=0.22, ge=0.0, le=1.0)
    sufficient_threshold: float = Field(default=0.32, ge=0.0, le=1.0)
    strong_relevance_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
