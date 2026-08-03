# Enterprise RAG

A Retrieval-Augmented Generation (RAG) application built using **LangChain**, **Ollama**, and **ChromaDB**.

This project demonstrates a traditional vector-based RAG pipeline that indexes PDF documents, performs semantic search using vector embeddings, and generates grounded responses using a local Large Language Model (LLM).

The project is designed with a modular, service-oriented architecture to make it easy to extend with more advanced retrieval techniques such as Parent Document Retrieval, Multi-Query Retrieval, Reranking, and Hybrid Search.

---

## Features

- PDF document ingestion
- Recursive text chunking
- Local embedding generation using Ollama
- Chroma vector database
- Semantic similarity search
- Retrieval-Augmented Generation (RAG)
- Source attribution
- Modular service-based architecture
- Local execution (No cloud LLM required)

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | LangChain |
| LLM | Ollama (llama3.2:3b) |
| Embedding Model | nomic-embed-text |
| Vector Database | ChromaDB |
| Document Loader | PyMuPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |

---

## Project Structure

```text
enterprise-rag/
│
├── app/
│   ├── cli/
│   │   ├── index_cli.py
│   │   ├── search_cli.py
│   │   └── rag_cli.py
│   │
│   ├── services/
│   │   ├── indexing_service.py
│   │   ├── retrieval_service.py
│   │   ├── rag_service.py
│   │   └── llm_service.py
│   │
│   └── config.py
│
├── chroma_db/
├── documents/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Architecture

```text
                    PDF Documents
                          │
                          ▼
               PyMuPDF Document Loader
                          │
                          ▼
        RecursiveCharacterTextSplitter
                          │
                          ▼
        Ollama Embedding Model
          (nomic-embed-text)
                          │
                          ▼
              Chroma Vector Database
                          │
                          ▼
             Semantic Similarity Search
                          │
                          ▼
              Retrieved Document Chunks
                          │
                          ▼
          Ollama LLM (llama3.2:3b)
                          │
                          ▼
            Grounded Answer with Sources
```

---

## Workflow

### Document Indexing

1. Load PDF documents.
2. Split documents into overlapping chunks.
3. Generate embeddings using Ollama.
4. Store embeddings in ChromaDB.

### Semantic Search

1. Convert the user query into an embedding.
2. Perform semantic similarity search.
3. Return the most relevant document chunks.

### Retrieval-Augmented Generation (RAG)

1. Retrieve relevant document chunks.
2. Build contextual prompt.
3. Generate a grounded answer using Ollama.
4. Display supporting document sources.

---

## Running the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

Application Menu:

```text
1. Index Documents
2. Semantic Search
3. Chat with Documents
4. Exit
```

---

## Example Questions

```text
What is the price of Explorer Pro 32L?

Which backpacks are water resistant?

What are the features of Weekend Backpacks?

List all backpacks above 30 liters.

What is the price of the HeadLamp Pro 500?
```

---

## Sample Dataset

To evaluate and demonstrate the RAG pipeline, this project currently uses publicly available **Contoso sample PDF documents** from **Microsoft Learn**.

These sample documents provide realistic product catalogs that are useful for testing document indexing, semantic retrieval, and question-answering workflows.

The documents are used solely for learning, experimentation, and validating the RAG pipeline.

---

## Current Limitations

The current implementation uses a traditional vector-based retrieval pipeline.

While it performs well for direct factual questions, there are still opportunities for improvement when handling:

- Category-based questions
- "List all" type questions
- Information distributed across multiple document sections
- Large structured documents
- Retrieval precision for semantically similar content

These limitations will be addressed as the project evolves with more advanced retrieval strategies.

---

## Roadmap

### Completed

- ✅ Traditional Vector RAG
- ✅ ChromaDB Integration
- ✅ Local LLM with Ollama
- ✅ Semantic Search
- ✅ Source Attribution
- ✅ Modular Service Architecture

### Planned Enhancements

- Parent Document Retrieval
- Multi-Query Retrieval
- Cross-Encoder Reranking
- Contextual Compression
- Hybrid Search
- Retrieval Evaluation
- Vectorless RAG
- Agentic RAG

---

## Learning Objectives

The primary goal of this project is to gain a deeper understanding of Retrieval-Augmented Generation (RAG) systems by building them from the ground up.

Rather than focusing only on generating answers, the project emphasizes understanding document ingestion, chunking strategies, vector indexing, semantic retrieval, prompt construction, and how retrieval quality impacts LLM-generated responses.

As the project evolves, additional retrieval techniques will be implemented and compared to understand their trade-offs and their impact on retrieval accuracy and answer quality.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.