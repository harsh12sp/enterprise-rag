from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.storage.file_system import LocalFileStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_RETRIEVAL_TOP_K,
    PARENT_STORE_DIRECTORY,
    PERSIST_DIRECTORY,
)


class ParentRetrievalService:
    """
    Uses smaller child chunks for precise semantic search and returns
    larger parent chunks for richer cross-page context.
    """

    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
        )

        self.vector_store = Chroma(
            collection_name=f"{COLLECTION_NAME}_children",
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        # Persist parent documents to disk instead of storing them
        # only in application memory.
        parent_store_path = Path(PARENT_STORE_DIRECTORY)
        parent_store_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        byte_store = LocalFileStore(
            parent_store_path,
        )

        self.parent_store = create_kv_docstore(
            byte_store,
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
            add_start_index=True,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
            add_start_index=True,
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.parent_store,
            parent_splitter=self.parent_splitter,
            child_splitter=self.child_splitter,
            child_metadata_fields=[
                "source",
                "total_pages",
                "document_type",
            ],
            search_kwargs={
                "k": PARENT_RETRIEVAL_TOP_K,
            },
        )

    def index_documents(
        self,
        documents: list[Document],
    ) -> dict[str, int]:
        """
        Creates parent documents, splits them into searchable child
        chunks, embeds the children, and persists both stores.
        """

        if not documents:
            raise ValueError(
                "No documents were provided for parent indexing."
            )

        self.retriever.add_documents(
            documents,
        )

        return {
            "documents": len(documents),
            "child_chunks": self.count_child_chunks(),
            "parent_documents": self.count_parent_documents(),
        }

    def search(
        self,
        query: str,
    ) -> list[Document]:
        """
        Searches child chunks and returns unique corresponding parents.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        retrieved_documents = self.retriever.invoke(
            query.strip(),
        )

        return self._remove_duplicate_documents(
            retrieved_documents,
        )

    @staticmethod
    def _remove_duplicate_documents(
        documents: list[Document],
    ) -> list[Document]:
        """
        Removes duplicate parent content while preserving retrieval order.
        """

        unique_documents: list[Document] = []
        seen_content: set[str] = set()

        for document in documents:
            normalized_content = (
                document.page_content.strip()
            )

            if normalized_content in seen_content:
                continue

            seen_content.add(normalized_content)
            unique_documents.append(document)

        return unique_documents

    def count_child_chunks(self) -> int:
        """
        Returns the number of child chunks stored in Chroma.
        """

        return self.vector_store._collection.count()

    def count_parent_documents(self) -> int:
        """
        Returns the number of parent documents stored on disk.
        """

        return sum(
            1 for _ in self.parent_store.yield_keys()
        )