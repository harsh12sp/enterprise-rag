from hashlib import sha256
from pathlib import Path
import json
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOCUMENTS_DIRECTORY,
    EMBEDDING_MODEL,
    PERSIST_DIRECTORY,
)


class LangChainIndexingService:
    """
    Loads PDF files, splits them into chunks, generates embeddings
    using Ollama, and stores the chunks in a persistent Chroma index.
    """

    def __init__(
        self,
        documents_directory: str = DOCUMENTS_DIRECTORY,
        persist_directory: str = PERSIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._validate_chunk_settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.documents_directory = Path(
            documents_directory
        )

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
        )

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
                add_start_index=True,
            )
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    @staticmethod
    def _validate_chunk_settings(
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """
        Validates text-splitting configuration.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )

    def load_documents(self) -> list[Document]:
        """
        Loads every PDF from the configured documents directory.

        PyMuPDFLoader creates one LangChain Document
        for each PDF page.
        """

        if not self.documents_directory.exists():
            raise FileNotFoundError(
                "Documents directory was not found: "
                f"{self.documents_directory.resolve()}"
            )

        pdf_files = self._get_pdf_files()

        if not pdf_files:
            raise FileNotFoundError(
                "No PDF files were found in: "
                f"{self.documents_directory.resolve()}"
            )

        documents: list[Document] = []

        for pdf_file in pdf_files:
            print(f"Loading: {pdf_file.name}")

            loader = PyMuPDFLoader(
                str(pdf_file)
            )

            pdf_documents = loader.load()

            for document in pdf_documents:
                self._add_document_metadata(
                    document=document,
                    source_name=pdf_file.name,
                )

            documents.extend(pdf_documents)

        return documents


    def load_documents_for_parent_retrieval(
        self,
    ) -> list[Document]:
        """
        Loads each PDF as one combined document.
        Page markers and page character boundaries are stored so
        retrieved parent chunks can later report accurate page ranges.
        """
        if not self.documents_directory.exists():
            raise FileNotFoundError(
                "Documents directory was not found: "
                f"{self.documents_directory.resolve()}"
            )
        pdf_files = self._get_pdf_files()
        if not pdf_files:
            raise FileNotFoundError(
                "No PDF files were found in: "
                f"{self.documents_directory.resolve()}"
            )
        combined_documents: list[Document] = []
        for pdf_file in pdf_files:
            print(
                f"Loading for parent retrieval: "
                f"{pdf_file.name}"
            )
            loader = PyMuPDFLoader(
                str(pdf_file)
            )
            page_documents = loader.load()
            page_sections: list[str] = []
            page_boundaries: list[dict[str, int]] = []
            separator = "\n\n"
            current_offset = 0
            for page_document in page_documents:
                zero_based_page = (
                    page_document.metadata.get(
                        "page",
                        0,
                    )
                )
                page_number = zero_based_page + 1
                page_content = (
                    page_document.page_content.strip()
                )
                page_section = (
                    f"--- PAGE {page_number} ---"
                    f"\n\n"
                    f"{page_content}"
                )
                # Account for the separator between pages.
                if page_sections:
                    current_offset += len(separator)
                page_start = current_offset
                page_sections.append(
                    page_section
                )
                current_offset += len(
                    page_section
                )
                page_end = current_offset
                page_boundaries.append(
                    {
                        "page_number": page_number,
                        "start": page_start,
                        "end": page_end,
                    }
                )
            combined_content = separator.join(
                page_sections
            )
            combined_document = Document(
                page_content=combined_content,
                metadata={
                    "source": pdf_file.name,
                    "total_pages": len(
                        page_documents
                    ),
                    "document_type": "pdf",
                    # Store as JSON so metadata remains serializable.
                    "page_boundaries": json.dumps(
                        page_boundaries
                    ),
                },
            )
            combined_documents.append(
                combined_document
            )
        return combined_documents

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Splits page documents into smaller,
        overlapping chunks.
        """

        if not documents:
            raise ValueError(
                "No documents were provided for splitting."
            )

        chunks = (
            self.text_splitter.split_documents(
                documents
            )
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk.metadata[
                "chunk_number"
            ] = chunk_number

        return chunks

    def index_documents(self) -> dict[str, int]:
        """
        Runs the complete document-indexing pipeline.
        """

        pdf_files = self._get_pdf_files()

        documents = self.load_documents()

        print(
            f"\nPages loaded: {len(documents)}"
        )

        chunks = self.split_documents(
            documents
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

        print(
            "Generating embeddings and "
            "storing chunks..."
        )

        chunk_ids = [
            self.create_chunk_id(chunk)
            for chunk in chunks
        ]

        self.vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        return {
            "pdf_files": len(pdf_files),
            "pages": len(documents),
            "chunks": len(chunks),
            "stored_chunks": self.count(),
        }

    @staticmethod
    def create_chunk_id(
        chunk: Document,
    ) -> str:
        """
        Creates a deterministic ID using source,
        page number, and chunk start position.
        """

        source = chunk.metadata.get(
            "source",
            "unknown",
        )

        page_number = chunk.metadata.get(
            "page_number",
            0,
        )

        start_index = chunk.metadata.get(
            "start_index",
            0,
        )

        raw_id = (
            f"{source}:"
            f"{page_number}:"
            f"{start_index}"
        )

        return sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

    def count(self) -> int:
        """
        Returns the number of records stored
        in the Chroma collection.
        """

        return (
            self.vector_store
            ._collection
            .count()
        )

    def _get_pdf_files(
        self,
    ) -> list[Path]:
        """
        Returns all PDF files from the documents directory.
        """

        return sorted(
            self.documents_directory.glob(
                "*.pdf"
            )
        )

    @staticmethod
    def _add_document_metadata(
        document: Document,
        source_name: str,
    ) -> None:
        """
        Adds normalized source and page metadata.
        """

        document.metadata[
            "source"
        ] = source_name

        zero_based_page = (
            document.metadata.get(
                "page",
                0,
            )
        )

        document.metadata[
            "page_number"
        ] = zero_based_page + 1