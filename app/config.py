import os

from dotenv import load_dotenv


# Load variables from the project-root .env file.
load_dotenv()


# ============================================================
# Documents
# ============================================================

DOCUMENTS_DIRECTORY = os.getenv(
    "DOCUMENTS_DIRECTORY",
    "documents",
)


# ============================================================
# Vector database
# ============================================================

PERSIST_DIRECTORY = os.getenv(
    "PERSIST_DIRECTORY",
    "chroma_db",
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "enterprise_documents",
)


# ============================================================
# Parent-document storage
# ============================================================

PARENT_STORE_DIRECTORY = os.getenv(
    "PARENT_STORE_DIRECTORY",
    "parent_store",
)


# ============================================================
# Embeddings
# ============================================================

# Embeddings remain local through Ollama.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
).strip()


# ============================================================
# LLM provider
# ============================================================

# Supported values:
# - groq
# - ollama
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "groq",
).strip().lower()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
).strip()

OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "llama3.2:3b",
).strip()

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0",
    )
)

LLM_MAX_RETRIES = int(
    os.getenv(
        "LLM_MAX_RETRIES",
        "2",
    )
)


# Compatibility with your existing main.py:
#
# llm_service = LLMService(
#     model_name=CHAT_MODEL
# )
CHAT_MODEL = (
    GROQ_MODEL
    if LLM_PROVIDER == "groq"
    else OLLAMA_CHAT_MODEL
)


# ============================================================
# Traditional RAG
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1200",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "4",
    )
)


# ============================================================
# Parent-document retrieval
# ============================================================

PARENT_CHUNK_SIZE = int(
    os.getenv(
        "PARENT_CHUNK_SIZE",
        "2000",
    )
)

PARENT_CHUNK_OVERLAP = int(
    os.getenv(
        "PARENT_CHUNK_OVERLAP",
        "200",
    )
)

CHILD_CHUNK_SIZE = int(
    os.getenv(
        "CHILD_CHUNK_SIZE",
        "400",
    )
)

CHILD_CHUNK_OVERLAP = int(
    os.getenv(
        "CHILD_CHUNK_OVERLAP",
        "80",
    )
)

# Parent retrieval depth by question type.
PARENT_NARROW_TOP_K = int(
    os.getenv(
        "PARENT_NARROW_TOP_K",
        "3",
    )
)

PARENT_DETAIL_TOP_K = int(
    os.getenv(
        "PARENT_DETAIL_TOP_K",
        "5",
    )
)

PARENT_AGGREGATION_TOP_K = int(
    os.getenv(
        "PARENT_AGGREGATION_TOP_K",
        "12",
    )
)

if PARENT_NARROW_TOP_K <= 0:
    raise ValueError(
        "PARENT_NARROW_TOP_K must be greater than zero."
    )

if PARENT_DETAIL_TOP_K <= 0:
    raise ValueError(
        "PARENT_DETAIL_TOP_K must be greater than zero."
    )

if PARENT_AGGREGATION_TOP_K <= 0:
    raise ValueError(
        "PARENT_AGGREGATION_TOP_K must be greater than zero."
    )

# Compatibility with older code.
PARENT_RETRIEVAL_TOP_K = PARENT_AGGREGATION_TOP_K


# ============================================================
# Validation
# ============================================================

if LLM_PROVIDER not in {"groq", "ollama"}:
    raise ValueError(
        "LLM_PROVIDER must be either 'groq' or 'ollama'."
    )

if not EMBEDDING_MODEL:
    raise ValueError(
        "EMBEDDING_MODEL cannot be empty."
    )

if not GROQ_MODEL:
    raise ValueError(
        "GROQ_MODEL cannot be empty."
    )

if not OLLAMA_CHAT_MODEL:
    raise ValueError(
        "OLLAMA_CHAT_MODEL cannot be empty."
    )

if LLM_MAX_RETRIES < 0:
    raise ValueError(
        "LLM_MAX_RETRIES cannot be negative."
    )

if CHUNK_SIZE <= 0:
    raise ValueError(
        "CHUNK_SIZE must be greater than zero."
    )

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be at least zero "
        "and smaller than CHUNK_SIZE."
    )

if TOP_K <= 0:
    raise ValueError(
        "TOP_K must be greater than zero."
    )

if PARENT_CHUNK_SIZE <= 0:
    raise ValueError(
        "PARENT_CHUNK_SIZE must be greater than zero."
    )

if not 0 <= PARENT_CHUNK_OVERLAP < PARENT_CHUNK_SIZE:
    raise ValueError(
        "PARENT_CHUNK_OVERLAP must be at least zero "
        "and smaller than PARENT_CHUNK_SIZE."
    )

if CHILD_CHUNK_SIZE <= 0:
    raise ValueError(
        "CHILD_CHUNK_SIZE must be greater than zero."
    )

if not 0 <= CHILD_CHUNK_OVERLAP < CHILD_CHUNK_SIZE:
    raise ValueError(
        "CHILD_CHUNK_OVERLAP must be at least zero "
        "and smaller than CHILD_CHUNK_SIZE."
    )

if PARENT_RETRIEVAL_TOP_K <= 0:
    raise ValueError(
        "PARENT_RETRIEVAL_TOP_K must be greater than zero."
    )