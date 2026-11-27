"""Sicklewise API: lets a website (or phone app) talk to the chatbot.

Run from the project folder:
    uvicorn main:app --reload --app-dir backend/app
Then open http://127.0.0.1:8000/docs to try it in your browser.
"""
import os

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import ask  # the chatbot brain in backend/app/ask.py

load_dotenv()

app = FastAPI(title="Sicklewise API", version="0.1")

# Allow the Next.js website (running on port 3000 during development) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the models once when the server starts, not on every question
embedder = SentenceTransformer(ask.EMBED_MODEL)
collection = chromadb.PersistentClient(path="data/chroma").get_collection("sicklewise")
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, examples=["What is acute chest syndrome?"])


class Source(BaseModel):
    title: str
    owner: str
    licence: str
    url: str
    kind: str          # "guidance" or "research"


class Answer(BaseModel):
    type: str          # "answer", "emergency", "crisis_support", "dose_refusal" or "dont_know"
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/ask", response_model=Answer)
def ask_question(q: Question):
    try:
        text, chunks = ask.answer(q.question, embedder, collection, llm)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"The assistant is unavailable right now: {e}")

    if text == ask.EMERGENCY_MESSAGE:
        kind = "emergency"
    elif text == ask.CRISIS_SUPPORT_MESSAGE:
        kind = "crisis_support"
    elif text == ask.DOSE_MESSAGE:
        kind = "dose_refusal"
    elif text == ask.DONT_KNOW_MESSAGE:
        kind = "dont_know"
    else:
        kind = "answer"

    # One entry per document, in the order they were used
    sources, seen = [], set()
    for c in chunks:
        m = c["meta"]
        if m["url"] in seen:
            continue
        seen.add(m["url"])
        sources.append(Source(title=m["title"], owner=m["owner"], licence=m["licence"],
                              url=m["url"], kind=m["kind"]))

    return Answer(type=kind, answer=text, sources=sources)