from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
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
    PERSIST_DIRECTORY,
)


class ParentRetrievalService:
    """
    Uses small child chunks for precise vector search and returns
    larger parent documents for richer LLM context.
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

        # Stores the larger parent documents.
        # This store is currently in memory and is cleared
        # when the application stops.
        self.parent_store = InMemoryStore()

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
        Creates parent documents, splits them into child chunks,
        embeds the child chunks, and stores the parents separately.
        """

        if not documents:
            raise ValueError(
                "No documents were provided for parent indexing."
            )

        self.retriever.add_documents(
            documents
        )

        return {
            "documents": len(documents),
            "child_chunks": self.count_child_chunks(),
        }

    def search(
        self,
        query: str,
    ) -> list[Document]:
        """
        Searches the child chunks and returns the corresponding
        larger parent documents.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        return self.retriever.invoke(
            query.strip()
        )

    def count_child_chunks(self) -> int:
        """
        Returns the number of child chunks stored in Chroma.
        """

        return self.vector_store._collection.count()