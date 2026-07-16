from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.document_loader import DocumentLoader
from app.chunker import TextChunker
from app.vector_store import VectorStore

def main():
    # Initialize services once
    llm = LLMService()
    loader = DocumentLoader()
    embedding_service = EmbeddingService()
    chunker = TextChunker(
        chunk_size=800,
        chunk_overlap=150
    )
    vector_store = VectorStore()

    print("1. Simple Chat")
    print("2. Load Full PDF Document")
    print("3. Load PDF Document in Chunks")
    print("4. Test Embeddings")
    print("5. Store Chunks in Vector Database")
    print("6. Search in Vector Database")
    print("7. Exit")

    choice = input("Enter your choice (1-7): ").strip()

    match choice:
        case "1":
            print("\n--- Running Simple Chat ---")

            question = input("Ask something: ").strip()

            if not question:
                print("Question cannot be empty.")
                return

            answer = llm.ask(question)
            print(f"\nAnswer: {answer}")

        case "2":
            print("\n--- Loading Full PDF Document ---")

            pdf_path = "documents/contoso-backpacks-guide.pdf"

            try:
                pages = loader.load_pdf(pdf_path)

                print(f"\nPages loaded: {len(pages)}")

                for page in pages:
                    print("\n----------------------------------------")
                    print(f"Source: {page['source']}")
                    print(f"Page: {page['page_number']}")
                    print()
                    print(page["text"])

            except Exception as error:
                print(f"Failed to load PDF: {error}")

        case "3":
            print("\n--- Loading PDF Document in Chunks ---")

            pdf_path = "documents/contoso-backpacks-guide.pdf"

            try:
                pages = loader.load_pdf(pdf_path)
                chunks = chunker.chunk_pages(pages)

                print(f"\nPages loaded: {len(pages)}")
                print(f"Chunks created: {len(chunks)}")

                for chunk in chunks[:3]:
                    print("\n----------------------------------------")
                    print(f"ID: {chunk['id']}")
                    print(f"Source: {chunk['source']}")
                    print(f"Page: {chunk['page_number']}")
                    print(f"Chunk: {chunk['chunk_number']}")
                    print()
                    print(chunk["text"])

            except Exception as error:
                print(f"Failed to process PDF: {error}")

        case "4":
            print("\n--- Testing Embeddings ---")

            sentences = [
                "I love pizza",
                "I like pizza",
                "The stock market crashed today"
            ]

            try:
                embeddings = embedding_service.create_embeddings(sentences)

                for sentence, vector in zip(sentences, embeddings):
                    print("\n----------------------------------------")
                    print(f"Sentence: {sentence}")
                    print(f"Dimensions: {len(vector)}")
                    print(f"First 5 values: {vector[:5]}")

            except Exception as error:
                print(f"Failed to generate embeddings: {error}")
  
        case "5":
            print("\n--- Storing Chunks in Vector Database ---")

            pdf_path = "documents/contoso-backpacks-guide.pdf"

            try:
                pages = loader.load_pdf(pdf_path)
                chunks = chunker.chunk_pages(pages)

                texts = [chunk["text"] for chunk in chunks]

                embeddings = embedding_service.create_embeddings(texts)

                ids = [chunk["id"] for chunk in chunks]

                metadatas = [
                    {
                        "source": chunk["source"],
                        "page_number": chunk["page_number"],
                        "chunk_number": chunk["chunk_number"]
                    }
                    for chunk in chunks
                ]

                vector_store.add_documents(
                    ids=ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                print(f"Chunks created: {len(chunks)}")
                print(f"Chunks stored: {vector_store.count()}")

            except Exception as error:
                print(f"Failed to store chunks: {error}")

                
        case "6":
            vector_search_example(vector_store, embedding_service)

        case "7":
            print("Exiting program. Goodbye!")

        case _:
            print("Invalid selection. Please choose between 1-7.")


def vector_search_example(vector_store: VectorStore, embedding_service: EmbeddingService):
    print("\n--- Searching in Vector Database ---")

    try:
        if vector_store.count() == 0:
            print("Vector database is empty. Run option 5 first.")
            return

        query = input("Enter your search query: ").strip()

        if not query:
            print("Search query cannot be empty.")
            return

        query_embedding = embedding_service.create_embedding(query)

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            print("No matching documents found.")
            return

        print(f"\nTop {len(documents)} results:")

        for index in range(len(documents)):
            document = documents[index]
            metadata = metadatas[index]
            distance = distances[index]

            print("\n----------------------------------------")
            print(f"Result {index + 1}")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Page: {metadata.get('page_number', 'Unknown')}")
            print(f"Chunk: {metadata.get('chunk_number', 'Unknown')}")
            print(f"Distance: {distance:.4f}")
            print()
            print(document)

    except Exception as error:
        print(f"Failed to perform search: {error}")

if __name__ == "__main__":
    main()
