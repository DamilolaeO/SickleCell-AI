"""Sicklewise: answer a question using ONLY the trusted knowledge base.

Order of checks for every question:
  1. Emergency check (fixed rules, no AI)   -> fixed urgent-care message
  2. Dose check (fixed rules, no AI)        -> polite refusal
  3. Search NHS / NICE / GOV.UK first, research papers added when they fit better
  4. Nothing relevant found                 -> "I don't know" (no AI guessing)
  5. AI writes the answer from the found chunks, with numbered citations
  6. Final check: block any answer that contains a dose

Run:  python backend/app/ask.py What is acute chest syndrome
"""
import os, re, sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------- Settings you can change ----------------
USE_RESEARCH = True          # switch research papers on or off
GUIDANCE_GOOD_ENOUGH = 0.80  # only skip research when NHS/NICE is a very strong match
MIN_MATCH = 0.55             # below this, nothing is relevant: say "I don't know"
MAX_PER_SOURCE = 2           # stop one document crowding out the rest
MAX_GUIDANCE, MAX_RESEARCH = 4, 2
LLM_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------- Fixed safety messages (never written by the AI) ----------------
EMERGENCY_MESSAGE = (
    "This could be an emergency.\n\n"
    "In the UK, call 999 now if you or the person you care for has severe pain, chest pain, "
    "difficulty breathing, a high temperature, signs of a stroke (face drooping, weakness on one "
    "side, slurred speech), a painful erection that will not go away, or is very drowsy or floppy.\n"
    "If you are not sure, call NHS 111 or your haematology team straight away.\n"
    "Outside the UK, call your local emergency number.\n\n"
    "Please do not wait for an answer from this app."
)
DOSE_MESSAGE = (
    "I can't give advice on how much medicine to take or give. The right amount depends on the "
    "person, the medicine and their other treatment.\n\n"
    "Please check the label or leaflet, or ask a pharmacist, your GP or your haematology team. "
    "If someone has taken too much of a medicine, call 111, or 999 if they are unwell."
)
DONT_KNOW_MESSAGE = (
    "I don't have trusted information on that yet, so I'd rather not guess. "
    "Your haematology team or GP is the best place to ask. "
    "The NHS sickle cell pages are also a good place to read more: "
    "https://www.nhs.uk/conditions/sickle-cell-disease/"
)

# Fixed NHS wording, added to any answer that touches urgent symptoms. Never written by the AI.
# Source: https://www.nhs.uk/conditions/sickle-cell-disease/ ("Call 999" section)
URGENT_NOTE = (
    "IMPORTANT: The NHS says to call 999 if you have sickle cell disease and:\n"
    "  - you have a high temperature\n"
    "  - you think you're having a sickle cell crisis and have severe pain, breathing "
    "difficulties or are being sick\n"
    "  - you have an erection that is painful or does not go away\n"
    "Do not drive yourself to A&E. Outside the UK, call your local emergency number."
)
URGENT_TOPICS = r"\b(fevers?|temperatures?|chest|breath\w*|crisis|crises|priapism|erections?|vomit\w*|being sick|acute chest syndrome|infections?)\b"

# Basic version. This list gets tested and expanded with many more phrasings later.
RED_FLAGS = (
    "chest pain", "chest is tight", "chest feels tight", "tight chest", "chest tight",
    "chest hurts", "chest is hurting", "pain in my chest", "pain in his chest", "pain in her chest",
    "can't breathe", "cant breathe", "cannot breathe", "struggling to breathe",
    "short of breath", "breathless", "difficulty breathing",
    "high temperature", "temperature of 38", "temperature of 39", "temperature of 40", "fever",
    "stroke", "face drooping", "face is drooping", "slurred", "speech is weird", "can't speak",
    "numb on one side", "weakness on one side", "can't move my arm", "can't move my leg",
    "fit", "seizure", "unconscious", "very drowsy", "floppy",
    "erection", "priapism",
    "swollen tummy", "swollen belly", "tummy is swollen", "belly is swollen",
    "really pale", "very pale",
    "crisis has started", "crisis is starting", "in crisis",
    "severe pain", "unbearable pain", "worst pain", "isn't touching it", "not touching the pain",
    "painkillers aren't working", "painkillers are not working",
    "keeps vomiting", "can't stop vomiting", "vomiting",
)
PERSONAL = (" i ", " i'm", " im ", " my ", " me ", " he ", " she ", " his ", " her ",
            " son", " daughter", " child", " baby", " kid", " husband", " wife", " partner")
EDUCATIONAL_STARTS = ("what is", "what are", "what's", "what causes", "what happens", "why",
                      "how", "explain", "tell me about", "define")
DOSE_PATTERNS = (
    r"how (much|many)\b.*\b(take|give|have)\b", r"\bdos(e|es|age|ing)\b",
    r"\bmax(imum)? (amount|daily)\b", r"\bhow often can (i|he|she|we)\b",
)
DOSE_IN_ANSWER = r"\b\d+(\.\d+)?\s?(mg|mcg|micrograms?|ml|g|grams?|tablets?|capsules?)\b|mg/kg"


def is_emergency(q):
    text = f" {q.lower()} "
    if not any(re.search(rf"\b{re.escape(flag)}(s|es)?\b", text) for flag in RED_FLAGS):
        return False
    educational = text.strip().startswith(EDUCATIONAL_STARTS) and not any(p in text for p in PERSONAL)
    return not educational  # when in doubt, treat it as an emergency


def asks_for_dose(q):
    return any(re.search(p, q.lower()) for p in DOSE_PATTERNS)


# ---------------- Search ----------------
def search(collection, q_emb, kind, limit):
    res = collection.query(query_embeddings=q_emb, n_results=12, where={"kind": kind})
    picked, per_source = [], {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - dist
        if score < MIN_MATCH:
            continue
        sid = meta["source_id"]
        if per_source.get(sid, 0) >= MAX_PER_SOURCE:
            continue
        per_source[sid] = per_source.get(sid, 0) + 1
        picked.append({"text": doc, "meta": meta, "score": score})
        if len(picked) == limit:
            break
    return picked


def retrieve(question, embedder, collection):
    q_emb = embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
    guidance = search(collection, q_emb, "guidance", MAX_GUIDANCE)
    best_guidance = max((c["score"] for c in guidance), default=0)
    if not USE_RESEARCH:
        return guidance
    research = search(collection, q_emb, "research", MAX_RESEARCH)
    best_research = max((c["score"] for c in research), default=0)
    # Add research when guidance is not a very strong match, or research fits the question better
    if best_guidance < GUIDANCE_GOOD_ENOUGH or best_research > best_guidance:
        return guidance + research
    return guidance


# ---------------- Answer ----------------
SYSTEM_PROMPT = """You are Sicklewise, a patient education assistant for people living with sickle cell disease and their families.

Rules:
- Answer ONLY using the numbered sources provided. If they do not answer the question, say you don't have trusted information on that.
- If the sources only partly answer the question, share what they do say, then be clear about what is not known.
- Cite sources in square brackets after each fact, like [1] or [2][3].
- If a fact comes from a source marked RESEARCH, start it with "Research suggests" and keep it cautious.
- Prefer sources marked GUIDANCE when they cover the point.
- Never give medicine doses, amounts or schedules. Never diagnose, triage, or interpret someone's own symptoms, results or records.
- Use plain, warm, UK English that a 12-year-old could follow. Explain medical words.
- Keep answers to about 150 to 250 words.
- End by suggesting they talk to their haematology team or GP if the question is about their own care.
- Never suggest waiting, watching for a day, or booking a routine GP appointment for a high temperature, chest pain, breathing problems, a painful erection or a severe crisis. These need urgent help."""


def build_prompt(question, chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        label = "RESEARCH" if c["meta"]["kind"] == "research" else "GUIDANCE"
        blocks.append(f"[{i}] ({label}) {c['meta']['title']}\n{c['text']}")
    return "SOURCES:\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


def answer(question, embedder, collection, llm):
    if is_emergency(question):
        return EMERGENCY_MESSAGE, []
    if asks_for_dose(question):
        return DOSE_MESSAGE, []

    chunks = retrieve(question, embedder, collection)
    if not chunks:
        return DONT_KNOW_MESSAGE, []

    reply = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=1500,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(question, chunks)}],
    )
    text = (reply.choices[0].message.content or "").strip()

    if re.search(DOSE_IN_ANSWER, text, flags=re.IGNORECASE):
        return DOSE_MESSAGE, chunks   # final safety net: never show a dose
    if not text:
        return DONT_KNOW_MESSAGE, []
    if re.search(URGENT_TOPICS, question + " " + text, flags=re.IGNORECASE):
        text += "\n\n" + URGENT_NOTE   # safety: always add the fixed NHS urgent advice
    return text, chunks


def format_sources(chunks):
    lines = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        lines.append(f"[{i}] {m['title']} | {m['owner']} | {m['licence']} | match {c['score']:.2f}\n    {m['url']}")
    return "\n".join(lines)


if __name__ == "__main__":
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "What is acute chest syndrome?"

    embedder = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path="data/chroma").get_collection("sicklewise")
    llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])

    text, chunks = answer(question, embedder, collection, llm)
    print(f"\nQUESTION: {question}\n\n{text}\n")
    if chunks:
        print("SOURCES\n" + format_sources(chunks))
