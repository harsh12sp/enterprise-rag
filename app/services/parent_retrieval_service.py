from pathlib import Path
import json
import re
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
    PARENT_AGGREGATION_TOP_K,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_DETAIL_TOP_K,
    PARENT_NARROW_TOP_K,
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
                "k": PARENT_NARROW_TOP_K,
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
        Selects retrieval depth based on the question type,
        retrieves parent documents, adds page metadata,
        and removes duplicates.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        cleaned_query = query.strip()

        retrieval_mode, top_k = (
            self._get_retrieval_strategy(
                cleaned_query
            )
        )

        # The application is currently a single-user CLI, so updating
        # search_kwargs before invoke is safe for this implementation.
        self.retriever.search_kwargs["k"] = top_k

        retrieved_documents = self.retriever.invoke(
            cleaned_query
        )

        unique_documents: list[Document] = []
        seen_documents: set[
            tuple[str, int, str]
        ] = set()

        for document in retrieved_documents:
            self._add_page_range_metadata(
                document
            )

            source = str(
                document.metadata.get(
                    "source",
                    "Unknown",
                )
            )

            start_index = int(
                document.metadata.get(
                    "start_index",
                    0,
                )
            )

            document_key = (
                source,
                start_index,
                document.page_content,
            )

            if document_key in seen_documents:
                continue

            seen_documents.add(
                document_key
            )

            # Useful for debugging and evaluation.
            document.metadata[
                "retrieval_mode"
            ] = retrieval_mode

            document.metadata[
                "retrieval_top_k"
            ] = top_k

            unique_documents.append(
                document
            )

        return unique_documents

    @staticmethod
    def _get_retrieval_strategy(
        query: str,
    ) -> tuple[str, int]:
        """
        Returns a retrieval mode and top-k value based on
        the apparent scope of the question.
        """

        normalized_query = " ".join(
            query.lower().split()
        )

        aggregation_phrases = (
            "list all",
            "show all",
            "find all",
            "compare all",
            "every product",
            "every tent",
            "all tents",
            "all products",
            "all accessories",
            "which tents",
            "which products",
            "which backpacks",
            "cheapest",
            "most expensive",
            "under $",
            "over $",
            "below $",
            "above $",
            "less than",
            "greater than",
            "how many",
        )

        detail_phrases = (
            "features",
            "specifications",
            "specs",
            "materials",
            "waterproof rating",
            "dimensions",
            "weight",
            "room configuration",
            "weather protection",
            "compare the features",
        )

        if any(
            phrase in normalized_query
            for phrase in aggregation_phrases
        ):
            return (
                "aggregation",
                PARENT_AGGREGATION_TOP_K,
            )

        if any(
            phrase in normalized_query
            for phrase in detail_phrases
        ):
            return (
                "detail",
                PARENT_DETAIL_TOP_K,
            )

        return (
            "narrow",
            PARENT_NARROW_TOP_K,
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

    @staticmethod
    def _add_page_range_metadata(
        document: Document,
    ) -> None:
        """
        Calculates the pages overlapped by a parent chunk using
        the chunk start index and stored PDF-page boundaries.
        """

        metadata = document.metadata

        raw_boundaries = metadata.get(
            "page_boundaries"
        )

        page_boundaries: list[dict[str, int]] = []

        if isinstance(raw_boundaries, str):
            try:
                page_boundaries = json.loads(
                    raw_boundaries
                )
            except json.JSONDecodeError:
                page_boundaries = []

        elif isinstance(raw_boundaries, list):
            page_boundaries = raw_boundaries

        chunk_start = int(
            metadata.get(
                "start_index",
                0,
            )
        )

        chunk_end = (
            chunk_start
            + len(document.page_content)
        )

        page_numbers: list[int] = []

        for boundary in page_boundaries:
            page_start = int(
                boundary.get(
                    "start",
                    0,
                )
            )

            page_end = int(
                boundary.get(
                    "end",
                    0,
                )
            )

            page_number = int(
                boundary.get(
                    "page_number",
                    0,
                )
            )

            # The chunk and page overlap when each starts before
            # the other one ends.
            overlaps_page = (
                chunk_start < page_end
                and chunk_end > page_start
            )

            if overlaps_page and page_number > 0:
                page_numbers.append(
                    page_number
                )

        # Fallback for older indexed documents that contain page
        # markers but do not contain page_boundaries metadata.
        if not page_numbers:
            marker_matches = re.findall(
                r"--- PAGE (\d+) ---",
                document.page_content,
            )

            page_numbers = [
                int(page)
                for page in marker_matches
            ]

        page_numbers = sorted(
            set(page_numbers)
        )

        if not page_numbers:
            metadata["start_page"] = None
            metadata["end_page"] = None
            metadata["page_numbers"] = ""
            return

        metadata["start_page"] = min(
            page_numbers
        )

        metadata["end_page"] = max(
            page_numbers
        )

        metadata["page_numbers"] = ",".join(
            str(page)
            for page in page_numbers
        )