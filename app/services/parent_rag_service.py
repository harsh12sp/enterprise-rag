import re
from typing import Any

from langchain_core.documents import Document

from app.services.llm_service import LLMService
from app.services.parent_retrieval_service import (
    ParentRetrievalService,
)


class ParentRAGService:
    """
    Generates answers using parent-document retrieval and
    returns only the source documents used by the LLM.
    """

    SOURCE_PATTERN = re.compile(
        r"\n?SOURCE_IDS:\s*([S\d,\s]+)\s*$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        parent_retrieval_service: ParentRetrievalService,
        llm_service: LLMService,
    ) -> None:
        self.parent_retrieval_service = (
            parent_retrieval_service
        )

        self.llm_service = llm_service

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Retrieves parent documents, generates the answer,
        and filters sources using the IDs returned by the LLM.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        cleaned_question = question.strip()

        documents = (
            self.parent_retrieval_service.search(
                cleaned_question
            )
        )

        if not documents:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the indexed documents."
                ),
                "documents": [],
                "retrieved_documents": [],
                "source_ids": [],
            }

        source_document_map = {
            f"S{index}": document
            for index, document in enumerate(
                documents,
                start=1,
            )
        }

        context = self.build_context(
            source_document_map
        )

        raw_answer = (
            self.llm_service.ask_with_context(
                question=cleaned_question,
                context=context,
            )
        )

        answer, source_ids = (
            self._extract_source_ids(
                raw_answer
            )
        )

        used_documents = (
            self._select_used_documents(
                source_ids=source_ids,
                source_document_map=(
                    source_document_map
                ),
            )
        )

        # Safe fallback:
        # If the model forgot to return valid source IDs,
        # preserve the original retrieved sources.
        if not used_documents:
            used_documents = documents

        return {
            "answer": answer,
            "documents": used_documents,
            "retrieved_documents": documents,
            "source_ids": source_ids,
        }

    @staticmethod
    def build_context(
        source_document_map: dict[
            str,
            Document,
        ],
    ) -> str:
        """
        Builds page-aware context with a unique ID for each
        retrieved parent document.
        """

        context_parts: list[str] = []

        for source_id, document in (
            source_document_map.items()
        ):
            metadata = document.metadata

            source = metadata.get(
                "source",
                "Unknown",
            )

            page_range = (
                ParentRAGService
                .format_page_range(metadata)
            )

            context_parts.append(
                f"""
<parent_document source_id="{source_id}">
Source ID: {source_id}
Source: {source}
Pages: {page_range}

Content:
{document.page_content}
</parent_document>
                """.strip()
            )

        return "\n\n".join(
            context_parts
        )

    @classmethod
    def _extract_source_ids(
        cls,
        raw_answer: str,
    ) -> tuple[str, list[str]]:
        """
        Removes the internal SOURCE_IDS line from the displayed
        answer and returns validated source IDs separately.
        """

        if not raw_answer:
            return "", []

        match = cls.SOURCE_PATTERN.search(
            raw_answer
        )

        if not match:
            return raw_answer.strip(), []

        raw_source_ids = match.group(1)

        source_ids: list[str] = []
        seen_ids: set[str] = set()

        for value in raw_source_ids.split(","):
            normalized_id = (
                value.strip().upper()
            )

            if not re.fullmatch(
                r"S\d+",
                normalized_id,
            ):
                continue

            if normalized_id in seen_ids:
                continue

            seen_ids.add(
                normalized_id
            )

            source_ids.append(
                normalized_id
            )

        cleaned_answer = (
            raw_answer[:match.start()]
            .rstrip()
        )

        return cleaned_answer, source_ids

    @staticmethod
    def _select_used_documents(
        source_ids: list[str],
        source_document_map: dict[
            str,
            Document,
        ],
    ) -> list[Document]:
        """
        Returns source documents in the order selected by
        the answer model.
        """

        selected_documents: list[Document] = []

        for source_id in source_ids:
            document = source_document_map.get(
                source_id
            )

            if document is None:
                continue

            selected_documents.append(
                document
            )

        return selected_documents

    @staticmethod
    def format_page_range(
        metadata: dict[str, Any],
    ) -> str:
        """
        Converts page metadata into a readable page range.
        """

        start_page = metadata.get(
            "start_page"
        )

        end_page = metadata.get(
            "end_page"
        )

        if start_page is None:
            return "Unknown"

        if end_page is None:
            return str(start_page)

        if start_page == end_page:
            return str(start_page)

        return (
            f"{start_page}-{end_page}"
        )