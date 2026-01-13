from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import requests

from .ingest import build_corpus

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "faiss.index"
META_PATH = BASE_DIR / "metadata.pkl"

EMBED_MODEL = "nomic-embed-text"  # modèle d'embedding Ollama


def get_embedding(text: str) -> List[float]:
    """
    Embedding via un modèle local Ollama.
    """
    url = "http://127.0.0.1:11434/api/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    emb = data["embedding"]  # liste de floats
    return np.array(emb, dtype="float32").tolist()


def build_index() -> None:
    corpus = build_corpus()
    if not corpus:
        raise RuntimeError("Corpus vide, vérifie le dossier data/.")

    print(f"{len(corpus)} chunks à indexer...")

    vectors = [get_embedding(item["text"]) for item in corpus]
    x = np.array(vectors).astype("float32")

    d = x.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(x)

    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("wb") as f:
        pickle.dump(corpus, f)

    print(f"Index FAISS sauvegardé dans {INDEX_PATH}")
    print(f"{index.ntotal} vecteurs indexés (dimension {d}).")


def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        corpus = pickle.load(f)
    return index, corpus


if __name__ == "__main__":
    build_index()
