class TextChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        chunks: list[dict] = []

        for page in pages:
            page_chunks = self._split_text(page["text"])

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunks.append(
                    {
                        "id": (
                            f'{page["source"]}'
                            f'-page-{page["page_number"]}'
                            f'-chunk-{chunk_index + 1}'
                        ),
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "chunk_number": chunk_index + 1,
                        "text": chunk_text,
                    }
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        normalized_text = " ".join(text.split())

        if not normalized_text:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(normalized_text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                last_space = normalized_text.rfind(" ", start, end)

                if last_space > start:
                    end = last_space

            chunk = normalized_text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - self.chunk_overlap

        return chunks