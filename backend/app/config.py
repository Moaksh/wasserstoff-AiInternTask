import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "data/uploads"))
PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", os.path.join(BASE_DIR, "data/processed"))
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER", os.path.join(BASE_DIR, "data/vector_db"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)