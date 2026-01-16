import os
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def list_documents() -> List[Path]:
    """
    Charge les chemins des docs
    """
    docs = []
    for name in os.listdir(DATA_DIR):
        if name.lower().endswith((".txt", ".md")):
            docs.append(DATA_DIR / name)
    return sorted(docs)


def read_text(path: Path) -> str:
    """
    Lit un fichier texte en UTF-8.
    """
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Découpe le texte en chunks de ~chunk_size mots, avec overlap mots
    de recouvrement entre deux chunks (exigence chunk_size ∈ [200,500] + overlap)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_corpus() -> List[Dict]:
    """
    Retourne une liste de dicts :
    {
        "doc_id": str,
        "chunk_id": int,
        "text": str,
        "source": str,
    }
    """
    corpus = []
    for i, path in enumerate(list_documents()):
        doc_id = f"doc_{i}"
        text = read_text(path)
        chunks = chunk_text(text)
        for j, ch in enumerate(chunks):
            corpus.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": j,
                    "text": ch,
                    "source": path.name,
                }
            )
    return corpus