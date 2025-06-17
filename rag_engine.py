
from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import os
from pymongo import MongoClient
from datetime import datetime
import asyncio
import time
import hashlib
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
    def __init__(self, data_folder="./data", urls=None):
        self.data_folder = data_folder
        self.urls = urls or []
        self.llm = OllamaLLM(model="mistral", temperature=0.2, streaming=True)
        self.embeddings = OllamaEmbeddings(model="mistral")

        # MongoDB
        self.mongo_uri = "mongodb://localhost:27017"
        self.db_name = "chatbot_db"
        self.collection_name = "chunks"
        self.mongo_client = MongoClient(self.mongo_uri)
        self.chunk_collection = self.mongo_client[self.db_name][self.collection_name]

        # Load FAISS
        self.retriever = self._load_data()

        # Kiểm tra retriever có null không
        if self.retriever is None:
            raise ValueError("❌ Retriever not loaded properly from FAISS.")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )


    def _load_data(self):
        # Load FAISS index
        vectorstore = FAISS.load_local(
            folder_path=self.data_folder,
            embeddings=self.embeddings,
            index_name="faiss_index/faiss_index",  # hoặc tên bạn đã lưu
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever()
        return retriever
    

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
