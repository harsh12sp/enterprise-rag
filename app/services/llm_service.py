from ollama import Client
from ollama import chat

class LLMService:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.client = Client(host=host)

    # LLMService class provides methods to interact with a language model for question answering and context-based responses. It initializes with a specified model and host, and includes methods to ask questions and provide context for more accurate answers.
    def ask(self, question: str) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        
        response = chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                )
        return response["message"]["content"]

    # The ask_with_context method allows the user to ask a question with an optional context. If context is provided, it constructs a prompt that instructs the model to answer only from the given context. If the answer is not found in the context, it instructs the model to respond with a specific message indicating that the information is not available. The method then sends this prompt to the language model and returns the generated response.
    def ask_with_context(self, question: str, context: str = "") -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if context:
            prompt = f"""
You are a document question-answering assistant.

Use only the provided context.

The user may make spelling mistakes. Correct obvious spelling mistakes internally when interpreting the question.

Do not infer unsupported facts.

If asked to list products:
- include only actual products
- exclude accessories
- exclude document titles
- distinguish between "waterproof" and "water-resistant"
- include a product only when the context explicitly says it is waterproof or has waterproof construction

If the answer is not available in the context, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}
""".strip()
        else:
            prompt = question.strip()

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]