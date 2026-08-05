from typing import Any

from langchain_core.documents import Document

from app.services.llm_service import LLMService
from app.services.parent_retrieval_service import (
    ParentRetrievalService,
)


class ParentRAGService:
    """
    Generates answers using parent-document retrieval.
    """

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
            }

        context = self.build_context(documents)

        answer = self.llm_service.ask_with_context(
            question=cleaned_question,
            context=context,
        )

        return {
            "answer": answer,
            "documents": documents,
        }

    @staticmethod
    def build_context(
        documents: list[Document],
    ) -> str:
        context_parts: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            context_parts.append(
                f"""
Parent document {index}

Source: {metadata.get("source", "Unknown")}
Start index: {metadata.get("start_index", "Unknown")}

Content:
{document.page_content}
                """.strip()
            )

        return "\n\n---\n\n".join(
            context_parts
        )