from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.llm.agent import agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.llm.rag import ask_docs, load_pdf_to_store
import os

load_dotenv()


app = FastAPI(
    title="CriptoBot",
    description="Este es un chatbot que puede responder el precio de las criptomonedas",
)

class ChatSchema(BaseModel):
    question: str
    thread_id: str = "default"

@app.post("/chatbot")
async def chat(request: ChatSchema):

    question = request.question
    thread_id = request.thread_id

    init_state = {"messages": [HumanMessage(content=question)]}

    config = {"configurable": {"thread_id": thread_id}}

    response = agent.invoke(init_state, config)
    return {"response": response["messages"][-1].content}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("./temp", exist_ok=True)
    file_path = f"./temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    await load_pdf_to_store(file_path)
    return {"status": f"PDF {file.filename} cargado y procesado"}


# Hacer preguntas a los documentos
@app.post("/ask_docs")
async def ask_documents(request: ChatSchema):
    answer = await ask_docs(request.question)
    return {"response": answer}