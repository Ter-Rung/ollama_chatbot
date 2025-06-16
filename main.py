from rag_engine import RAGEngine
from fastapi import FastAPI, Form, Request
import os
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio


app = FastAPI()  


# URLs bạn muốn load từ web (có thể thêm nhiều)
web_urls = [
    "https://superwal.io/",
    "https://superwal.io/about",
]

# Khởi tạo engine
rag = RAGEngine(urls=web_urls)

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
