import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================
# 🔧 Step 1: Configure LM Studio local endpoint
# ============================================
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"  # placeholder key

# ============================================
# 📄 Step 2: Load and process PDF
# ============================================
loader = PDFPlumberLoader("Basic_Home_Remedies.pdf")
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(docs)
print(f"✅ Created {len(documents)} chunks.")

# ============================================
# 🔍 Step 3: Build FAISS vector database
# ============================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ============================================
# 🤖 Step 4: Initialize LM Studio LLM
# ============================================
llm = ChatOpenAI(
    model="tinyllama-1.1b-chat-v1.0",
    temperature=0.7,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=60,
)

# ============================================
# 🧠 Step 5: Create prompt and chain
# ============================================
prompt = ChatPromptTemplate.from_template("""
You are a health expert assistant.
Use the provided context to answer the question clearly and accurately.
If the answer is not in the context, say "The information is not available in the provided context."

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

# ============================================
# 💬 Step 6: Hybrid Q&A (PDF + General Chat)
# ============================================
while True:
    query = input("\n❓ Enter your question (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # Step 1: Try answering from PDF context
    answer_from_pdf = rag_chain.invoke(query)

    # Step 2: If not found → fallback to direct chat
    if "not available in the provided context" in answer_from_pdf.lower():
        print("\n📚 Info not in PDF. Asking LM Studio directly...")
        response = llm.invoke(query)
        print("🧠 General Answer:", response.content)
    else:
        print("💡 Context-based Answer:", answer_from_pdf)
