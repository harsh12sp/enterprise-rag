from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class RetrievalService:
    """
    Retrieves relevant document chunks from the existing
    persistent Chroma vector database.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "enterprise_documents",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.embeddings = OllamaEmbeddings(
            model=embedding_model
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_configuration={
                            "hnsw": {
                                "space": "cosine"
                            }
                        },
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        Performs semantic search and returns matching documents
        with their Chroma distance scores.
        """

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.count() == 0:
            raise ValueError(
                "The vector database is empty. "
                "Run document indexing first."
            )

        return self.vector_store.similarity_search_with_score(
            query=query.strip(),
            k=top_k,
        )

    def get_retriever(
        self,
        top_k: int = 3,
    ):
        """
        Returns a LangChain retriever.

        This will be used later by the RAG service.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
            },
        )

    def count(self) -> int:
        """
        Returns the number of chunks stored in Chroma.
        """

        return self.vector_store._collection.count()