from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from config import MONGO_URI, DB_NAME, COLLECTION_NAME, DATA_FOLDER, URLS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from dotenv import load_dotenv
from connect_db import CheckDB
import datetime
import os
import hashlib

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")
URLS = os.getenv("URLS", "").split(",")

# === Khởi tạo MongoDB ===
db_checker = CheckDB()
# Truy cập collection
collection = db_checker.collection


# === Tạo splitter ===
def process_urls():
    print("🌐 Đang tải nội dung từ URL...")
    documents = []

    for url in URLS:
        if not url.strip():
            continue

        try:
            loader = WebBaseLoader(url.strip())
            raw_docs = loader.load()

            for doc in raw_docs:
                doc.metadata["source"] = url
                doc.metadata["title"] = url.split("/")[-1].capitalize()

            documents.extend(raw_docs)
            print(f"✅ Đã tải từ: {url}")

        except Exception as e:
            print(f"❌ Lỗi khi tải {url}: {e}")

    return documents

def process_local_files():
    print("📂 Đang tải nội dung từ file trong thư mục:", DATA_FOLDER)
    documents = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_FOLDER, filename)
            loader = TextLoader(path)
            raw_docs = loader.load()

            for doc in raw_docs:
                doc.metadata["source"] = path
                doc.metadata["title"] = os.path.splitext(filename)[0]

            documents.extend(raw_docs)
            print(f"✅ Đã tải từ file: {filename}")

    return documents

def chunk_and_store(docs):
    print("🔪 Đang chunk văn bản...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"📦 Tổng số chunk: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["created_at"] = datetime.datetime.utcnow().isoformat()

        collection.insert_one({
            "content": chunk.page_content,
            **chunk.metadata
        })

    print("✅ Đã lưu chunk vào MongoDB!")

if __name__ == "__main__":
    docs = []
    docs += process_urls()
    docs += process_local_files()
    
    if docs:
        chunk_and_store(docs)
    else:
        print("⚠️ Không tìm thấy dữ liệu để xử lý.")