"""Command-line Q&A over your documents.

Usage:
    python task2_cli.py sample.txt
"""
import sys

from rag import answer, build_retriever, get_sources, load_documents


def main() -> None:
    paths = sys.argv[1:] or ["sample.txt"]
    print(f"Indexing: {', '.join(paths)} ...")
    retriever = build_retriever(load_documents(paths))
    print("Ready. Ask a question (type 'exit' to quit).\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        print("Bot:", answer(q, retriever))
        pages = {d.metadata.get("page", "-") for d in get_sources(q, retriever)}
        print(f"(sources: page(s) {sorted(map(str, pages))})\n")


if __name__ == "__main__":
    main()
