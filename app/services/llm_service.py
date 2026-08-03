from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class LLMService:
    """
    Handles communication with the local Ollama language model.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        temperature: float = 0,
    ) -> None:
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

        self.rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a document question-answering assistant.

Answer only from the provided context.

Follow these rules:

1. Do not use outside knowledge.
2. Do not invent missing information.
3. Ignore retrieved content that is unrelated to the question.
4. If the answer is not available in the context, say:
   "I could not find that information in the provided documents."
5. When the question asks about a category containing multiple products,
   identify every matching product found in the context.
6. Group category answers by product name.
7. Do not merge the features of different products into one list.
8. Do not stop after describing only the first matching product.
9. When answering a price, capacity, weight, or other direct factual
   question, respond clearly and briefly.
10. Do not mention the retrieved chunks unless the user asks about them.
                    """.strip(),
                ),
                (
                    "human",
                    """
Context:

{context}

Question:

{question}

Answer:
                    """.strip(),
                ),
            ]
        )

    def ask(self, question: str) -> str:
        """
        Sends a general question to the Ollama model.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        response = self.llm.invoke(question.strip())

        return response.content.strip()

    def ask_with_context(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Answers a question using only the supplied document context.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty.")

        messages = self.rag_prompt.format_messages(
            question=question.strip(),
            context=context.strip(),
        )

        response = self.llm.invoke(messages)

        return response.content.strip()