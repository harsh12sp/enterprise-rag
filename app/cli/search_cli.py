from app.services.retrieval_service import RetrievalService


def run_semantic_search(
    retrieval_service: RetrievalService,
) -> None:
    print("\n--- Semantic Search ---")
    print("Type 'back' to return to the main menu.")

    try:
        if retrieval_service.count() == 0:
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
        query = input("\nSearch: ").strip()

        if query.lower() in {"back", "exit"}:
            return

        if not query:
            print("Search query cannot be empty.")
            continue

        try:
            results = retrieval_service.search(
                query=query,
                top_k=3,
            )

            if not results:
                print("No matching documents found.")
                continue

            for rank, (
                document,
                distance,
            ) in enumerate(
                results,
                start=1,
            ):
                metadata = document.metadata

                print("\n" + "-" * 70)
                print(f"Result: {rank}")
                print(
                    f"Source: "
                    f"{metadata.get('source', 'Unknown')}"
                )
                print(
                    f"Page: "
                    f"{metadata.get('page_number', 'Unknown')}"
                )
                print(
                    f"Chunk: "
                    f"{metadata.get('chunk_number', 'Unknown')}"
                )
                print(f"Distance: {distance:.4f}")
                print()
                print(document.page_content)

        except Exception as error:
            print(f"Search failed: {error}")