import os
from pathlib import Path


APP_TITLE = "Document Q&A API with Theme Identification Wassertoff AI intern task"
APP_DESCRIPTION = "Upload PDFs, manage them, query content, and identify themes using AI. (Modular Structure)"


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:

    raise ValueError("CRITICAL: GOOGLE_API_KEY environment variable not set.")


EMBEDDING_MODEL_NAME = "models/embedding-001"
GENERATION_MODEL_NAME = "gemma-3-1b-it"
LLM_TEMPERATURE = 0.4


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH_ROOT = BACKEND_DIR / "data"

UPLOADED_PDFS_PATH = DATA_PATH_ROOT / "uploaded_pdfs"
CHROMA_DB_PATH = DATA_PATH_ROOT / "chroma_db"


os.makedirs(UPLOADED_PDFS_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)


CHROMA_COLLECTION_NAME = "document_collection"


TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200


DEFAULT_QUERY_TOP_K = 5
MAX_QUERY_TOP_K = 15


LOG_LEVEL = "DEBUG"
