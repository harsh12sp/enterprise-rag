from app.services.langchain_indexing_service import (
    LangChainIndexingService,
)


def run_indexing(
    indexing_service: LangChainIndexingService,
) -> None:
    print("\n--- Document Indexing ---")

    try:
        result = indexing_service.index_documents()

        print("\nIndexing completed successfully.")
        print("-" * 70)
        print(
            f"PDF files processed: "
            f"{result['pdf_files']}"
        )
        print(f"Pages loaded: {result['pages']}")
        print(f"Chunks created: {result['chunks']}")
        print(
            f"Chunks stored: "
            f"{result['stored_chunks']}"
        )

    except Exception as error:
        print(f"Indexing failed: {error}")