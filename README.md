# AI PDF Chat Assistant (with Voice Output)

This project is a FastAPI-based PDF Question–Answering system that uses:
- Local LLM (LM Studio)
- LangChain RAG (PDF retrieval)
- Frontend HTML UI
- Voice Output with ON/OFF toggle
- Replay button for last spoken answer
- PDF Download for Chat

## Features
- Upload PDF and ask questions from its content.
- If answer is missing in PDF → fallback to LLM general answer.
- Output voice ON/OFF button.
- Replay answer button.
- Download full chat as PDF.

## Installation
pip install -r requirements.txt

## Run Backend
uvicorn app:app --reload

## Run LM Studio
Set Base URL → http://localhost:1234/v1
API Key → anything (e.g., lm-studio)

## Open Frontend
http://127.0.0.1:8000
