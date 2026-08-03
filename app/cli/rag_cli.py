from app.services.rag_service import RAGService


def run_rag_chat(
    rag_service: RAGService,
) -> None:
    print("\n--- Chat with Documents ---")
    print("Type 'back' to return to the main menu.")

    try:
        if rag_service.retrieval_service.count() == 0:
            print(
                "The vector database is empty. "
                "Index the documents first."
            )
            return

    except Exception as error:
        print(
            "Could not connect to the vector database: "
            f"{error}"
        )
        return

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"back", "exit"}:
            return

        if not question:
            print("Question cannot be empty.")
            continue

        try:
            result = rag_service.ask(question)

            print("\nAnswer:")
            print("-" * 70)
            print(result["answer"])

            retrieved_results = result["results"]

            if not retrieved_results:
                continue

            print("\nRetrieved Sources:")

            for document, distance in retrieved_results:
                metadata = document.metadata

                print(
                    f"- "
                    f"{metadata.get('source', 'Unknown')}, "
                    f"page "
                    f"{metadata.get('page_number', 'Unknown')}, "
                    f"chunk "
                    f"{metadata.get('chunk_number', 'Unknown')}, "
                    f"distance {distance:.4f}"
                )

        except Exception as error:
            print(f"RAG request failed: {error}")