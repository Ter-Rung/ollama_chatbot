# from fastapi import FastAPI, Form
# from chain import RAGEngine

# app = FastAPI()
# rag = RAGEngine()

# @app.get("/")
# def root():
#     return {"message": "Welcome to the SuperWAL RAG chatbot 🚀"}

# @app.post("/ask")
# def ask_question(question: str = Form(...)):
#     answer = rag.ask(question)
#     return {"question": question, "answer": answer}

from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
# from starlette.concurrency import run_in_threadpool
from langchain_community.vectorstores import FAISS
from fastapi.concurrency import run_in_threadpool
import os
import asyncio
import time

app = FastAPI()

# from langchain.chains import LLMChain
# from langchain.chains.question_answering import load_qa_chain

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


# Khởi tạo RAG chỉ 1 lần
class RAGEngine:
    def __init__(self, data_folder="./data"):
        self.data_folder = data_folder
        self.llm = OllamaLLM(model="mistral", temperature=0.2, streaming=True) 
        self.embeddings = OllamaEmbeddings(model="mistral")
        self.retriever = self._load_data()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )



    def _load_data(self):
        index_path = os.path.join(self.data_folder, "faiss_index")
        if os.path.exists(index_path):
            return FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            ).as_retriever(search_kwargs={"k": 3})
        
        docs = []
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_folder, filename))
                docs.extend(loader.load())

        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        db = FAISS.from_documents(texts, self.embeddings)
        db.save_local(index_path)
        return db.as_retriever(search_kwargs={"k": 3})

    def run_sync(self, question: str) -> str:
        return self.qa_chain.run(question)



    async def ask(self, question: str) -> str:
        try:
            # Chạy LLM trong threadpool để không block event loop, giới hạn timeout 15s
            result = await asyncio.wait_for(
                run_in_threadpool(self.run_sync, question),
                timeout=150
            )
            return result
        except asyncio.TimeoutError:
            return "⏱️ Xin lỗi, hệ thống phản hồi chậm. Vui lòng thử lại sau."
        except Exception as e:
            # Log lỗi chi tiết nếu cần
            print(f"❌ Lỗi khi gọi LLM: {e}")
            return "⚠️ Đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại sau."

# Khởi tạo 1 instance duy nhất
rag = RAGEngine()

@app.get("/")
def root():
    return {"message": "Welcome to the SuperWAL RAG chatbot 🚀"}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    answer = await rag.ask(question)
    return {"question": question, "answer": answer}

@app.post("/chat-stream")
async def chat_stream(request: Request):
    try:
        # 👇 Giả sử client gửi dạng JSON
        data = await request.json()
        question = data.get("question", "")

        async def generate():
            yield "🤔 Đang xử lý...\n\n"
            await asyncio.sleep(0.5)
            answer = await rag.ask(question)
            for chunk in answer.split():
                yield chunk + " "
                await asyncio.sleep(0.05)

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        async def error_stream():
            yield "⚠️ Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau."
        return StreamingResponse(error_stream(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok", "model": "mistral", "backend": "Ollama"}
