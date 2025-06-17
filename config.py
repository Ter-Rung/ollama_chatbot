# config.py
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chunks")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")

urls_raw = os.getenv("URLS", "")
URLS = [u.strip() for u in urls_raw.split(",") if u.strip()]
