from __future__ import annotations

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.rag_chat import llm_no_rag, search_topk, format_context, llm_rag_answer  # type: ignore


def main() -> None:
    print("Pose tes questions (tape 'quit' pour sortir).")

    while True:
        question = input("\nQuestion : ")
        if not question or question.lower() in {"quit", "exit"}:
            break

        print("\n=== no-RAG ===\n")
        print(llm_no_rag(question))

        print("\n=== RAG ===\n")
        topk = search_topk(question, k=4)
        context = format_context(topk)
        print(llm_rag_answer(question, context))


if __name__ == "__main__":
    main()
