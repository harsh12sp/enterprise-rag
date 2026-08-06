from typing import Any

from app.services.parent_rag_service import (
    ParentRAGService,
)


def run_parent_rag_chat(
    parent_rag_service: ParentRAGService,
) -> None:
    print("\n--- Parent Document RAG ---")
    print(
        "Type 'back' to return to the main menu."
    )

    while True:
        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() in {
            "back",
            "exit",
        }:
            return

        if not question:
            print(
                "Question cannot be empty."
            )
            continue

        try:
            result = parent_rag_service.ask(
                question
            )

            print("\nAnswer:")
            print("-" * 70)
            print(
                result["answer"]
            )

            documents = result[
                "documents"
            ]

            if not documents:
                continue

            print("\nRetrieved Sources:")

            displayed_sources: set[
                tuple[str, Any, Any]
            ] = set()

            for document in documents:
                metadata = document.metadata

                source = metadata.get(
                    "source",
                    "Unknown",
                )

                start_page = metadata.get(
                    "start_page"
                )

                end_page = metadata.get(
                    "end_page"
                )

                source_key = (
                    str(source),
                    start_page,
                    end_page,
                )

                if source_key in displayed_sources:
                    continue

                displayed_sources.add(
                    source_key
                )

                page_text = (
                    _format_page_range(
                        start_page=start_page,
                        end_page=end_page,
                    )
                )

                print(
                    f"- {source}, {page_text}"
                )

        except Exception as error:
            print(
                f"Parent RAG request failed: "
                f"{error}"
            )


def _format_page_range(
    start_page: Any,
    end_page: Any,
) -> str:
    if start_page is None:
        return "pages unknown"

    if end_page is None:
        return f"page {start_page}"

    if start_page == end_page:
        return f"page {start_page}"

    return (
        f"pages {start_page}-{end_page}"
    )