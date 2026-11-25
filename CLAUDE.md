Sicklewise

Sicklewise is a free education app for people living with sickle cell disease and their parents and carers. It has two parts:

An education chatbot that answers only from trusted, openly licensed sources (NHS, NICE, GOV.UK, US government sites and CC BY research papers), with citations.
A pain journal where users record pain episodes and see factual summaries to take to appointments.

It does NOT diagnose, triage, give medicine doses or interpret a user's own symptoms, results or journal entries. See docs/intended-purpose.md and docs/scope.md.

About me

I'm a health data scientist, not a full-time software engineer. Explain changes in plain English, keep code simple, and tell me exactly which commands to run and from which folder. I'm on Windows using PowerShell.

Stack
Chatbot: Python 3.13, FastAPI, ChromaDB, sentence-transformers (BAAI/bge-small-en-v1.5), Groq API with openai/gpt-oss-20b
Website: Next.js (App Router only), TypeScript, Tailwind, shadcn/ui (Base UI, Nova preset), in frontend/
Journal backend (to build): Express + MongoDB Atlas (UK/EU region)
Project layout
backend/app/ask.py: the chatbot brain (safety checks, search, answer, citations)
backend/app/main.py: FastAPI server (POST /ask, GET /health)
backend/rag/: scripts to download sources, download papers and build the search index
data/sources.csv, data/papers_candidates.csv: source lists (data/raw/ and data/chroma/ are private and git-ignored)
evaluation/test_set.csv: 91 test questions with expected result types
evaluation/batch_test.py: runs the test set and scores PASS/FAIL
frontend/: Next.js website
Commands (from the project root unless stated)
Activate Python env: .venv\Scripts\Activate.ps1
Run the API: uvicorn main:app --reload --app-dir backend/app (test page: http://127.0.0.1:8000/docs)
Ask one question: python backend\app\ask.py <question>
Safety tests only: python evaluation\batch_test.py emergency dose off_topic
Full test set: python evaluation\batch_test.py
Rebuild search index: python backend\rag\build_index.py
Website (from frontend/): npm run dev (http://localhost:3000)
Safety rules: never break these
Emergency messages, dose refusals, "I don't know" and the NHS urgent note are fixed text in ask.py. The AI must never write them.
Emergency detection has two layers (word rules + AI yes/no check). Either one is enough. Never weaken either.
After ANY change to ask.py, run the safety tests. Missed emergencies must stay at 0. If a change lowers any safety score, undo it.
The AI answers only from retrieved sources, with citations. Uncited answers become "I don't know".
Never show medicine doses.
The website must show reply types differently: emergency as a prominent alert with 999/111 guidance, dose_refusal and dont_know with signposting, answer with numbered sources.
Journal data is special category health data (UK GDPR). Never send journal data to the AI model. Summaries of journal entries are calculated in code, not by AI.
Secrets
API keys live in .env (git-ignored). Never print, copy, commit or move .env or its contents.
Style
UK English (haematology, anaemia, hydroxycarbamide, A&E).
Plain language a 12-year-old could follow in anything users see.
No em dashes in user-facing text.