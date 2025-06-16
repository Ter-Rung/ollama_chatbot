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
import asyncio
import time
import hashlib


#Tạo prompt
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

#
class RAGEngine:
    #Hàm quét file txt xuất ra prompt và gửi cho mistral
    def __init__(self, data_folder="./data", urls=None):
        self.data_folder = data_folder
        self.llm = OllamaLLM(model="mistral", temperature=0.2, streaming=True) 
        self.embeddings = OllamaEmbeddings(model="mistral")
        self.urls = urls or []
        self.retriever = self._load_data()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

    #Hàm băm urls
    def _hash_urls(self):
        hash_object = hashlib.md5("".join(self.urls).encode())
        return hash_object.hexdigest()

    #Hàm load file và lưu vào faiss để mỗi lần gọi lệnh không phải load lại file txt 
    def _load_data(self):
        index_path = os.path.join(self.data_folder, "faiss_index")
        index_file = os.path.join(index_path, "faiss_index")
        pkl_file = os.path.join(index_path, "faiss_index.pkl")

        if os.path.exists(index_file) and os.path.exists(pkl_file):
            return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 3})

        docs = []

        # Load local .txt files
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_folder, filename))
                docs.extend(loader.load())

        # Load from web if URLs provided
        if self.urls:
            web_loader = WebBaseLoader(self.urls)
            docs.extend(web_loader.load())

        # Chunking
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)

        # FAISS vector store
        db = FAISS.from_documents(texts, self.embeddings)
        db.save_local(index_path)
        return db.as_retriever(search_kwargs={"k": 3})

    def run_sync(self, question: str) -> str:
        return self.qa_chain.run(question)

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
