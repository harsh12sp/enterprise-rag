from pathlib import Path

import fitz


class DocumentLoader:
    def load_pdf(self, file_path: str) -> list[dict]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {path.name}")

        pages: list[dict] = []

        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                if not text:
                    continue

                pages.append(
                    {
                        "source": path.name,
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )

        return pages