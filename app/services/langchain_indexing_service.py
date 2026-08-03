from hashlib import sha256
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainIndexingService:
    """
    Loads PDF files, splits them into chunks, generates embeddings
    using Ollama, and stores the chunks in a persistent Chroma index.
    """

    def __init__(
        self,
        documents_directory: str = "documents",
        persist_directory: str = "chroma_db",
        collection_name: str = "enterprise_documents",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.documents_directory = Path(documents_directory)
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embeddings = OllamaEmbeddings(
            model=embedding_model
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def load_documents(self) -> List[Document]:
        """
        Loads every PDF from the configured documents directory.

        PyMuPDFLoader creates one LangChain Document per PDF page.
        """

        if not self.documents_directory.exists():
            raise FileNotFoundError(
                f"Documents directory was not found: "
                f"{self.documents_directory.resolve()}"
            )

        pdf_files = sorted(self.documents_directory.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files were found in "
                f"{self.documents_directory.resolve()}"
            )

        documents: List[Document] = []

        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file.name}")

            loader = PyMuPDFLoader(str(pdf_file))
            pdf_documents = loader.load()

            for document in pdf_documents:
                # Replace the full path with a cleaner filename.
                document.metadata["source"] = pdf_file.name

                # PyMuPDFLoader uses zero-based page numbers.
                zero_based_page = document.metadata.get("page", 0)
                document.metadata["page_number"] = zero_based_page + 1

            documents.extend(pdf_documents)

        return documents

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Splits page documents into smaller overlapping chunks.
        """

        if not documents:
            raise ValueError("No documents were provided for splitting.")

        chunks = self.text_splitter.split_documents(documents)

        for chunk_number, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_number"] = chunk_number

        return chunks

    @staticmethod
    def create_chunk_id(chunk: Document) -> str:
        """
        Creates a deterministic ID from source, page and chunk position.

        Deterministic IDs prevent duplicate records when the same
        documents are indexed again.
        """

        source = chunk.metadata.get("source", "unknown")
        page_number = chunk.metadata.get("page_number", 0)
        start_index = chunk.metadata.get("start_index", 0)

        raw_id = f"{source}:{page_number}:{start_index}"

        return sha256(raw_id.encode("utf-8")).hexdigest()

    def index_documents(self) -> dict:
        """
        Runs the complete vector-indexing pipeline.
        """

        documents = self.load_documents()

        print(f"\nPages loaded: {len(documents)}")

        chunks = self.split_documents(documents)

        print(f"Chunks created: {len(chunks)}")
        print("Generating embeddings and storing chunks...")

        chunk_ids = [
            self.create_chunk_id(chunk)
            for chunk in chunks
        ]

        self.vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        return {
            "pdf_files": len(
                list(self.documents_directory.glob("*.pdf"))
            ),
            "pages": len(documents),
            "chunks": len(chunks),
            "stored_chunks": self.count(),
        }

    def count(self) -> int:
        """
        Returns the total number of records in the Chroma collection.
        """

        return self.vector_store._collection.count()