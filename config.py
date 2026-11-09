import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
# RAG_DATA_DIR: read a corpus in place (e.g. the 236-PDF RAG-Multi-Corpus folder)
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data" / "documents"))
DATA_GLOB = os.getenv("RAG_DATA_GLOB", "*")  # e.g. "*.pdf": skip a corpus folder's LICENSE/SOURCE.md
STORAGE_DIR = BASE_DIR / "storage" / "chroma"

# Enbeddings
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIAMENSION = 384  # bge small output size

# Chunking
CHUNK_SIZE = 512  # tokens (approximate — we use character count)
CHUNK_OVERLAP = 64  # overlap between adjacent chunks


# Retrival
TOP_K = 5  # how many chunks to retrieve

# Chroma
CHROMA_COLLECTION = "personal_kb"

# LLM
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"        # Use till it expires
# OPENROUTER_MODEL = "inclusionai/ling-3.0-flash:free"            # second option
