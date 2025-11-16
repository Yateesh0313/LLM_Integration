from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# =====================================================
# FASTAPI SETUP
# =====================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# =====================================================
# LM STUDIO SETUP
# =====================================================
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"

llm = ChatOpenAI(
    model="tinyllama-1.1b-chat-v1.0",
    temperature=0.7,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = None
retriever = None

# =====================================================
# PDF UPLOAD
# =====================================================
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    global retriever

    with open(file.filename, "wb") as f:
        f.write(await file.read())

    loader = PDFPlumberLoader(file.filename)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    return {"message": f"📄 PDF Loaded Successfully ({len(chunks)} chunks)"}

# =====================================================
# CHAT ENDPOINT
# =====================================================
class Query(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(query: Query):
    global retriever

    # If NO PDF uploaded → LLM general answer
    if retriever is None:
        output = llm.invoke(f"Answer clearly in 2–3 sentences:\n{query.question}")
        return {"answer": output.content}

    # Try answering using PDF
    prompt = ChatPromptTemplate.from_template("""
    Use ONLY the provided PDF context to answer the question.
    Answer in 2–3 simple sentences.
    If answer is not in PDF, reply exactly: NO_DATA

    Context:
    {context}

    Question:
    {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    pdf_answer = chain.invoke(query.question)

    # Fallback when PDF doesn’t contain answer
    if "NO_DATA" in pdf_answer:
        general = llm.invoke(f"Answer clearly in 2–3 sentences:\n{query.question}")
        return {"answer": general.content}

    return {"answer": pdf_answer}

# =====================================================
# FRONTEND SERVING
# =====================================================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse("llm.html")
