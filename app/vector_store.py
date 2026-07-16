import chromadb


class VectorStore:

    def __init__(self):
        # What is PersistentClient?
        # It creates a local database. path chroma_db
        self.client = chromadb.PersistentClient(
            path="chroma_db"
        )
        # What is a Collection?
        # Think of it like a SQL table.
        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add_documents(
        self,
        ids,
        documents,
        embeddings,
        metadatas
    ):
        # use upcert instead of add to avoid duplicate ID
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
   
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    def count(self):
        return self.collection.count()    
    
    def clear(self):
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents"
        )