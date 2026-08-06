import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.config import (
    CHAT_MODEL,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


class LLMService:
    """
    Generates answers using Groq or local Ollama.

    Ollama continues to provide local embeddings elsewhere in the
    application. This class controls only chat and RAG answer generation.

    The model_name parameter remains compatible with the existing main.py:

        llm_service = LLMService(model_name=CHAT_MODEL)
    """

    def __init__(
        self,
        model_name: str = CHAT_MODEL,
        provider: str = LLM_PROVIDER,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        self.provider = provider.strip().lower()
        self.model_name = model_name.strip()

        if not self.model_name:
            raise ValueError(
                "Model name cannot be empty."
            )

        self.llm = self._create_llm(
            temperature=temperature,
        )

        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an enterprise document question-answering assistant.

Answer using only the supplied retrieved context.

GENERAL GROUNDING RULES

1. Examine every retrieved parent document before answering.
2. Do not use outside knowledge.
3. Do not invent products, prices, specifications, categories,
   capacities, or other information.
4. Ignore context unrelated to the user's question.
5. Treat every retrieved parent document as an independent source block.
6. Never combine attributes belonging to different products.
7. Keep each product's name, SKU, category, capacity, price,
   weight, and other specifications together.
8. If the context does not contain the answer, respond exactly:
   "I could not find that information in the provided documents."

LIST AND AGGREGATION RULES

9. For list, comparison, filtering, or aggregation questions,
   inspect every retrieved parent document.
10. Identify every product that satisfies all conditions in the question.
11. Search all represented catalog sections and categories.
12. Include products from specialty, four-season, backpacking,
    family, solo, or other sections only when they satisfy every
    condition in the user's question.
13. Do not stop after finding the first matching products.
14. Remove duplicate products from the final answer.
15. Before answering, compare the final result against every
    retrieved parent document and verify that no matching product
    was omitted.
16. Do not include a product merely because it is mentioned in
    the retrieved context.

STRICT FILTERING RULES

17. Apply category filters using explicit document text only.
18. Do not infer a product category from its name, features,
    waterproof rating, intended use, or marketing description.
19. Do not assume that a tent is four-season merely because it is
    suitable for alpine use, has a high waterproof rating, or is
    designed for difficult conditions.
20. For specialty or four-season questions, include only products
    explicitly categorized or explicitly described as specialty
    or four-season.
21. Apply numeric filters exactly.
22. For a price limit such as "under $400", include only products
    whose explicit price is below $400.
23. For capacity filters such as "exactly 2 persons", include only
    products explicitly showing a capacity of 2 persons.
24. A product must satisfy every requested filter to appear in
    the final result.
25. Do not include products with missing required values.
26. Do not include non-matching products merely to explain why
    they were excluded.
27. Do not classify a product using words such as "implies",
    "probably", "likely", "possibly", or "may".

TABLE FORMATTING RULES

28. For product-list and comparison questions, return a Markdown table.
29. Every table row must represent exactly one matching product.
30. Never put explanatory sentences, exclusion reasons, or notes
    inside a table row.
31. Do not create placeholder rows such as:
    "No other products were found."
32. Do not put excluded products inside the result table.
33. If an exclusion note is useful, place one short sentence after
    the table.
34. Include only columns requested by the user, plus product name
    and SKU when available.
35. Use "Not provided" only when the user requests a field and the
    matching product is valid, but that field is genuinely absent.
36. Do not create blank cells when a value is unavailable.

PAGE AND CHUNK RULES

37. Do not claim that information crosses pages or chunks unless
    the supplied context or metadata explicitly proves it.
38. A start index alone does not prove that a product crosses pages.
39. Do not guess page boundaries from text position.
40. If page or chunk boundaries cannot be verified, clearly state:
    "The supplied metadata does not provide enough information to
    verify the exact page or chunk boundaries."

FINAL VERIFICATION

Before returning the answer, silently verify:

- Every included product satisfies every requested condition.
- No matching product in the context was omitted.
- No non-matching product was included.
- No attributes were mixed between products.
- No category was inferred from marketing language.
- Every table row contains one valid matching product.
- No explanation was inserted as a table row.
                    """.strip(),
                ),
                (
                    "human",
                    """
Retrieved parent-document context:

{context}

User question:

{question}

Return a complete, precise, and grounded answer using only the
retrieved context.
                    """.strip(),
                ),
            ]
        )

    def _create_llm(
        self,
        temperature: float,
    ) -> Any:
        """
        Creates the configured Groq or Ollama chat model.
        """

        if self.provider == "groq":
            api_key = os.getenv(
                "GROQ_API_KEY"
            )

            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY is missing. "
                    "Add it to the project-root .env file."
                )

            return ChatGroq(
                model=self.model_name,
                temperature=temperature,
                max_retries=LLM_MAX_RETRIES,
            )

        if self.provider == "ollama":
            return ChatOllama(
                model=self.model_name,
                temperature=temperature,
            )

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}. "
            "Use 'groq' or 'ollama'."
        )

    def ask(
        self,
        question: str,
    ) -> str:
        """
        Sends a general question directly to the configured model.
        """

        cleaned_question = self._validate_text(
            value=question,
            field_name="Question",
        )

        response = self.llm.invoke(
            cleaned_question
        )

        return self._extract_response_text(
            response.content
        )

    def ask_with_context(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Answers a question using only retrieved RAG context.
        """

        cleaned_question = self._validate_text(
            value=question,
            field_name="Question",
        )

        cleaned_context = self._validate_text(
            value=context,
            field_name="Context",
        )

        messages = self.rag_prompt.format_messages(
            question=cleaned_question,
            context=cleaned_context,
        )

        response = self.llm.invoke(
            messages
        )

        return self._extract_response_text(
            response.content
        )

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
    ) -> str:
        """
        Validates and trims question or context input.
        """

        if not value or not value.strip():
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return value.strip()

    @staticmethod
    def _extract_response_text(
        content: Any,
    ) -> str:
        """
        Converts Groq or Ollama response content into plain text.

        Most responses contain a string, but this also supports
        block-based response formats.
        """

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(
                        block
                    )
                    continue

                if isinstance(block, dict):
                    text = block.get(
                        "text"
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

                    continue

                text_parts.append(
                    str(block)
                )

            return "\n".join(
                text_parts
            ).strip()

        return str(content).strip()