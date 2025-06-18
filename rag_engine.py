
# from fastapi import FastAPI, Form, Request
# from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from langchain_community.document_loaders import TextLoader, WebBaseLoader
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
# from bs4 import BeautifulSoup
import os
from pymongo import MongoClient
from datetime import datetime
import asyncio
# import time
# import hashlib
from langchain.schema import Document

# Tạo prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Bạn là một trợ lý AI thông minh. Dựa trên nội dung sau, hãy trả lời câu hỏi một cách ngắn gọn, chính xác và đầy đủ.

Nội dung:
{context}

Câu hỏi:
{question}
"""
)




class RAGEngine:
    def __init__(self, data_folder="./data", urls=None, save_path="faiss_index"):
            self.data_folder = data_folder
            self.urls = urls or []
            self.index_folder = save_path
            self.index_path = os.path.join(save_path, "faiss_index")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = OllamaLLM(model="mistral", temperature=0.2, streaming=True)

            # MongoDB
            self.mongo_uri = "mongodb://localhost:27017"
            self.db_name = "chatbot_db"
            self.collection_name = "chunks"
            self.mongo_client = MongoClient(self.mongo_uri)
            self.chunk_collection = self.mongo_client[self.db_name][self.collection_name]

            # Nếu index chưa có thì tạo
            if not os.path.exists(self.index_path + ".faiss"):
                self.build_vectorstore()

            # Load FAISS
            self.retriever = self._load_data()

            if self.retriever is None:
                raise ValueError("❌ Retriever not loaded properly from FAISS.")

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True
            )

    def _load_data(self):
        vectorstore = FAISS.load_local(
            folder_path=self.index_folder,
            embeddings=self.embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever()

    def build_vectorstore(self):
        if os.path.exists(self.index_path + ".faiss"):
            print("✅ FAISS index đã tồn tại. Bỏ qua bước build.")
            return

        print("⚙️  Đang build FAISS index...")

        # Xóa dữ liệu MongoDB cũ nếu có
        self.chunk_collection.delete_many({})

        docs = []
        for file in os.listdir(self.data_folder):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_folder, file), encoding="utf-8")
                raw_docs = loader.load()
                for doc in raw_docs:
                    doc.metadata["source"] = file
                docs.extend(raw_docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        now = datetime.utcnow().isoformat()
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunk_id"] = i
            doc.metadata["created_at"] = now
            filename = doc.metadata.get("source", "")
            doc.metadata["title"] = filename.replace(".txt", "")
            self.chunk_collection.insert_one({
                "chunk_index": i,
                "chunk_id": i,
                "content": doc.page_content,
                "source": filename,
                "title": doc.metadata["title"],
                "created_at": now
            })

        db = FAISS.from_documents(split_docs, self.embeddings)
        db.save_local(self.index_folder)
        print("✅ Đã tạo và lưu FAISS index tại:", self.index_folder)
    

    def _get_metadata_from_mongo(self, source, chunk_index):
        doc = self.chunk_collection.find_one({
            "source": source,
            "chunk_index": chunk_index
        })
        return doc

    def run_sync(self, question: str) -> str:
        result = self.qa_chain({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])

        # Map metadata lại từ MongoDB
        enriched_sources = []
        for doc in sources:
            original_meta = doc.metadata
            source = original_meta.get("source")
            chunk_index = original_meta.get("chunk_id") or original_meta.get("chunk_index") or 0

            mongo_meta = self._get_metadata_from_mongo(source, chunk_index)
            if mongo_meta:
                enriched_sources.append(f"- 📄 `{mongo_meta.get('title', '')}` (chunk #{mongo_meta.get('chunk_index')})")
            else:
                enriched_sources.append(f"- 📄 `{source}` (chunk #{chunk_index})")

        metadata_display = "\n".join(enriched_sources)
        return f"{answer}\n\n📚 Nguồn tham khảo:\n{metadata_display}"
    
    async def ask(self, question: str) -> str:
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(self.run_sync, question),
                timeout=150
            )
            return result
        except asyncio.TimeoutError:
            return "⏱️ Xin lỗi, hệ thống phản hồi chậm. Vui lòng thử lại sau."
        except Exception as e:
            print(f"❌ Lỗi khi gọi LLM: {e}")
            return "⚠️ Đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại sau."
