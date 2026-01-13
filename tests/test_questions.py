from __future__ import annotations

from pathlib import Path
from typing import List

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.rag_chat import llm_no_rag, search_topk, format_context, llm_rag_answer


QUESTIONS: List[str] = [
    "Que se passe-t-il si j'arrive en retard à un examen ?",
    "Comment justifier une absence à un examen ?",
    "Ai-je le droit d'utiliser une calculatrice pendant les examens ?",
    "Comment sont communiqués les résultats d'examen et puis-je contester une note ?",
    "Puis-je utiliser un justificatif d'absence pour tous les examens de l'année ? (hors doc)",
    "Puis-je passer un examen en ligne depuis chez moi ? (hors doc)",
]


def run_comparison() -> None:
    rows = []

    for i, q in enumerate(QUESTIONS, start=1):
        print(f"=== Q{i} ===")
        print(q)
        print("-------------")

        no_rag = llm_no_rag(q)

        topk = search_topk(q, k=4)
        context = format_context(topk)
        rag = llm_rag_answer(q, context)

        rows.append((f"Q{i}", no_rag, rag))

    lines = []
    lines.append("| Question | no-RAG (résumé + pb) | RAG (résumé + sources) |")
    lines.append("|---------|-----------------------|------------------------|")
    for qid, no_rag_ans, rag_ans in rows:
        no_rag_cell = no_rag_ans.replace("\n", " ")
        rag_cell = rag_ans.replace("\n", " ")
        lines.append(f"| {qid} | {no_rag_cell} | {rag_cell} |")

    table_md = "\n".join(lines)

    out_path = Path(__file__).resolve().parent / "comparaison_noRAG_vs_RAG.md"
    out_path.write_text(table_md, encoding="utf-8")

    print(f"\nTableau Markdown complet écrit dans : {out_path}")


if __name__ == "__main__":
    run_comparison()
