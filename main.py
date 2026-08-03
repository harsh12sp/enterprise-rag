from app.cli.index_cli import run_indexing
from app.cli.rag_cli import run_rag_chat
from app.cli.search_cli import run_semantic_search
from app.services.langchain_indexing_service import (
    LangChainIndexingService,
)
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService


DOCUMENTS_DIRECTORY = "documents"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "enterprise_documents"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 4


def main() -> None:
    indexing_service = LangChainIndexingService(
        documents_directory=DOCUMENTS_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    retrieval_service = RetrievalService(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
    )

    llm_service = LLMService(
        model_name=CHAT_MODEL,
        temperature=0,
    )

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        top_k=RETRIEVAL_TOP_K,
    )

    while True:
        print_menu()

        choice = input(
            "\nEnter your choice (1-4): "
        ).strip()

        match choice:
            case "1":
                run_indexing(indexing_service)

            case "2":
                run_semantic_search(retrieval_service)

            case "3":
                run_rag_chat(rag_service)

            case "4":
                print("\nExiting program. Goodbye!")
                break

            case _:
                print(
                    "\nInvalid selection. "
                    "Choose a number between 1 and 4."
                )


def print_menu() -> None:
    print("\n" + "=" * 70)
    print("ENTERPRISE RAG")
    print("=" * 70)
    print("1. Index Documents")
    print("2. Semantic Search")
    print("3. Chat with Documents")
    print("4. Exit")


if __name__ == "__main__":
    main()