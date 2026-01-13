from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import pickle

import faiss
import numpy as np
import requests

from .ingest import build_corpus

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "faiss.index"
META_PATH = BASE_DIR / "metadata.pkl"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:latest"

def embed_text(text: str) -> List[float]:
    """
    Calcule un embedding pour une chaîne de texte avec le modèle EMBED_MODEL.
    """
    url = "http://127.0.0.1:11434/api/embeddings"
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    emb = data["embedding"]
    return np.array(emb, dtype="float32").tolist()

def build_index() -> None:
    """
    1) récupère les chunks
    2) calcule un embedding pour chaque chunk
    3) construit un index FAISS
    4) sauvegarde index + métadonnées (corpus)
    """
    corpus = build_corpus()
    if not corpus:
        raise RuntimeError("Corpus vide, vérifie le dossier data/.")

    print(f"{len(corpus)} chunks à indexer...")

    vectors = [embed_text(item["text"]) for item in corpus]
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
    """
    Charge l'index et les métadonnées pour la phase de retrieval.[file:85]
    """
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        corpus = pickle.load(f)
    return index, corpus

def search_topk(question: str, k: int = 4) -> List[Dict]:
    """
    Implémente le retrieval top-k :
      - calcule l'embedding de la question
      - récupère les k meilleurs chunks
      - renvoie score, source, extrait du chunk (consignes 7.4).[file:85]
    """
    index, corpus = load_index()

    q_vec = np.array([embed_text(question)]).astype("float32")
    distances, indices = index.search(q_vec, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        meta = corpus[int(idx)]
        meta = {
            **meta,
            "score": float(dist),
        }
        results.append(meta)
    return results

def format_context(chunks: List[Dict]) -> str:
    """
    Formate le contexte avec citations [source:chunk_id]
    """
    parts = []
    for c in chunks:
        citation = f"[{c['source']}:{c['chunk_id']}]"
        parts.append(f"{c['text']}\n{citation}")
    return "\n\n---\n\n".join(parts)

def llm_rag_answer(question: str, context: str) -> str:
    """
    Génération RAG avec consignes :
      - répondre UNIQUEMENT à partir du contexte
      - sinon dire 'Je ne sais pas.'
      - citer les sources sous la forme [doc:chunk_id].[file:82][file:85]
    """
    system_prompt = (
        "Tu es un assistant factuel. Tu dois répondre UNIQUEMENT à partir du CONTEXTE fourni. "
        "Si l'information n'est pas dans le contexte, réponds exactement : \"Je ne sais pas\". "
        "Tu dois citer tes sources en fin de phrase dans la forme [source:chunk_id]."
    )

    prompt = (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXTE:\n{context}\n\n"
        f"USER:\nQuestion: {question}\n\n"
        "ASSISTANT:\nDonne une réponse de 5 à 10 lignes avec des citations."
    )

    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def rag_chat_loop() -> None:
    index, _ = load_index()
    print(f"Index chargé avec {index.ntotal} vecteurs.")
    print("Pose une question sur les examens (ou 'quit' pour sortir).")

    while True:
        question = input("> ")
        if not question or question.lower() in {"quit", "exit"}:
            break

        topk = search_topk(question, k=4)
        context = format_context(topk)
        answer = llm_rag_answer(question, context)

        print("\n=== RÉPONSE RAG ===\n")
        print(answer)
        print("\n===================\n")

def llm_no_rag(question: str) -> str:
    """
    Mode no-RAG : on pose directement la question au LLM sans contexte.[file:82]
    """
    prompt = (
        "Tu es un assistant qui répond sans avoir accès à des documents externes. "
        "Réponds à la question suivante le mieux possible, même si tu n'as pas toutes les informations.\n\n"
        f"Question: {question}\n\n"
        "Réponse :"
    )

    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")