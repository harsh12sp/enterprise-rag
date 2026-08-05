from app.services.parent_rag_service import (
    ParentRAGService,
)


def run_parent_rag_chat(
    parent_rag_service: ParentRAGService,
) -> None:
    print("\n--- Parent Document RAG ---")
    print("Type 'back' to return to the main menu.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"back", "exit"}:
            return

        if not question:
            print("Question cannot be empty.")
            continue

        try:
            result = parent_rag_service.ask(
                question
            )

            print("\nAnswer:")
            print("-" * 70)
            print(result["answer"])

            documents = result["documents"]

            if not documents:
                continue

            print("\nParent Sources:")

            for document in documents:
                metadata = document.metadata

                print(
                    f"- "
                    f"{metadata.get('source', 'Unknown')}, "
                    f"start index "
                    f"{metadata.get('start_index', 'Unknown')}"
                )

        except Exception as error:
            print(
                f"Parent RAG request failed: {error}"
            )