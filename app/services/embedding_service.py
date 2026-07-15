from ollama import Client


class EmbeddingService:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def create_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        response = self.client.embed(
            model=self.model,
            input=text.strip(),
        )

        return response["embeddings"][0]

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text.strip()]

        if not clean_texts:
            raise ValueError("At least one non-empty text is required.")

        response = self.client.embed(
            model=self.model,
            input=clean_texts,
        )

        return response["embeddings"]