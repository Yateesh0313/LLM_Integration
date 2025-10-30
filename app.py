from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + embeddings globally
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenAI(
    model="tinyllama-1.1b-chat-v1.0",
    temperature=0.7,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=60,
)

vectorstore = None
retriever = None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global retriever
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    loader = PDFPlumberLoader(file.filename)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    vectorstore_local = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore_local.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return {"message": f"✅ Loaded {len(documents)} chunks from {file.filename}"}

class Query(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(query: Query):
    if retriever is None:
        return {"answer": "⚠️ Please upload a PDF first."}

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful medical assistant.
    Use the provided context to answer clearly.
    If not found, say: "Information not found in the PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query.question)
    if "not found" in answer.lower():
        response = llm.invoke(query.question)
        return {"answer": response.content}
    else:
        return {"answer": answer}
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse("llm.html")
