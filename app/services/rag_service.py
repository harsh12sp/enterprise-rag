from typing import Any

from langchain_core.documents import Document

from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService


class RAGService:
    """
    Coordinates retrieval, context preparation,
    answer generation, and source collection.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.top_k = top_k

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Retrieves relevant chunks and generates a grounded answer.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        cleaned_question = question.strip()

        results = self.retrieval_service.search(
            query=cleaned_question,
            top_k=self.top_k,
        )

        if not results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the indexed documents."
                ),
                "results": [],
            }

        documents = [
            document
            for document, _ in results
        ]

        context = self.build_context(
            documents
        )

        answer = self.llm_service.ask_with_context(
            question=cleaned_question,
            context=context,
        )

        return {
            "answer": answer,
            "results": results,
        }

    @staticmethod
    def build_context(
        documents: list[Document],
    ) -> str:
        """
        Converts LangChain documents into a structured
        context block for the LLM.
        """

        if not documents:
            return ""

        context_parts: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page_number",
                "Unknown",
            )

            chunk = metadata.get(
                "chunk_number",
                "Unknown",
            )

            context_part = f"""
Retrieved document {index}

Source: {source}
Page: {page}
Chunk: {chunk}

Content:
{document.page_content}
            """.strip()

            context_parts.append(
                context_part
            )

        return "\n\n---\n\n".join(
            context_parts
        )