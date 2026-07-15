from ollama import chat


class LLMService:

    def __init__(self, model="llama3.2:3b"):
        self.model = model

    def ask(self, question: str):

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