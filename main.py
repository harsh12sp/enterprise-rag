from app.cli.index_cli import run_indexing
from app.cli.parent_rag_cli import run_parent_rag_chat
from app.cli.rag_cli import run_rag_chat
from app.cli.search_cli import run_semantic_search

from app.services.indexing_service import (
    LangChainIndexingService,
)
from app.services.llm_service import LLMService
from app.services.parent_rag_service import (
    ParentRAGService,
)
from app.services.parent_retrieval_service import (
    ParentRetrievalService,
)
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService

from app.config import (
    CHAT_MODEL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
    TOP_K,
)


def main() -> None:
    """
    Initializes all application services once and keeps
    the menu running until the user exits.
    """

    indexing_service = LangChainIndexingService()

    retrieval_service = RetrievalService(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    )

    parent_retrieval_service = ParentRetrievalService()

    llm_service = LLMService(
        model_name=CHAT_MODEL,
    )

    traditional_rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        top_k=TOP_K,
    )

    parent_rag_service = ParentRAGService(
        parent_retrieval_service=parent_retrieval_service,
        llm_service=llm_service,
    )

    while True:
        print_menu()

        choice = input(
            "\nEnter your choice (1-6): "
        ).strip()

        match choice:
            case "1":
                run_indexing(
                    indexing_service
                )

            case "2":
                run_semantic_search(
                    retrieval_service
                )

            case "3":
                run_rag_chat(
                    traditional_rag_service
                )

            case "4":
                run_parent_retrieval_test(
                    indexing_service=indexing_service,
                    parent_retrieval_service=(
                        parent_retrieval_service
                    ),
                )

            case "5":
                run_parent_rag_chat(
                    parent_rag_service
                )

            case "6":
                print(
                    "\nExiting program. Goodbye!"
                )
                break

            case _:
                print(
                    "\nInvalid selection. "
                    "Choose a number between 1 and 6."
                )


def print_menu() -> None:
    print("\n" + "=" * 70)
    print("ENTERPRISE RAG")
    print("=" * 70)
    print("1. Index Documents")
    print("2. Semantic Search")
    print("3. Traditional RAG")
    print("4. Index and Test Parent Document Retrieval")
    print("5. Parent Document RAG")
    print("6. Exit")


def run_parent_retrieval_test(
    indexing_service: LangChainIndexingService,
    parent_retrieval_service: ParentRetrievalService,
) -> None:
    """
    Loads PDFs as combined documents, creates parent and
    child chunks, and allows parent retrieval testing.

    The parent store is in memory, so this option must be
    executed before Parent Document RAG in the same run.
    """

    print(
        "\n--- Parent Document Retrieval ---"
    )

    try:
        documents = (
            indexing_service
            .load_documents_for_parent_retrieval()
        )

        result = (
            parent_retrieval_service
            .index_documents(documents)
        )

        print(
            "\nParent indexing completed."
        )
        print("-" * 70)
        print(
            f"Combined PDF documents: "
            f"{result['documents']}"
        )
        print(
            f"Child chunks stored: "
            f"{result['child_chunks']}"
        )

    except Exception as error:
        print(
            f"Parent indexing failed: {error}"
        )
        return

    while True:
        query = input(
            "\nParent search query "
            "(type 'back' to return): "
        ).strip()

        if query.lower() in {
            "back",
            "exit",
        }:
            print(
                "Returning to main menu."
            )
            return

        if not query:
            print(
                "Search query cannot be empty."
            )
            continue

        try:
            parent_documents = (
                parent_retrieval_service
                .search(query)
            )

            if not parent_documents:
                print(
                    "No parent documents found."
                )
                continue

            print(
                f"\nParent documents returned: "
                f"{len(parent_documents)}"
            )

            for index, document in enumerate(
                parent_documents,
                start=1,
            ):
                metadata = document.metadata

                print("\n" + "=" * 80)
                print(
                    f"Parent result: {index}"
                )
                print("-" * 80)
                print(
                    f"Source: "
                    f"{metadata.get('source', 'Unknown')}"
                )
                print(
                    f"Start index: "
                    f"{metadata.get('start_index', 'Unknown')}"
                )
                print(
                    f"Total pages: "
                    f"{metadata.get('total_pages', 'Unknown')}"
                )
                print()
                print(document.page_content)

        except Exception as error:
            print(
                f"Parent search failed: {error}"
            )


if __name__ == "__main__":
    main()