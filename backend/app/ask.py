"""Sicklewise: answer a question using ONLY the trusted knowledge base.

Order of checks for every question:
  1. Emergency check (fixed rules, no AI)   -> fixed urgent-care message
  2. Dose check (fixed rules, no AI)        -> polite refusal
  3. Search NHS / NICE / GOV.UK first, research papers added when they fit better
  4. Nothing relevant found                 -> "I don't know" (no AI guessing)
  5. AI writes the answer from the found chunks, with numbered citations
     (asked again once if it forgets citations; if still none -> "I don't know")
  6. Final checks: block doses, remove behind-the-scenes wording,
     keep only the sources actually cited, numbered to match the answer

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
    "side, slurred speech), a sudden severe headache or changes in vision, a painful erection that "
    "will not go away, or is very drowsy or floppy.\n"
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
# In the AI's answer, only real symptom advice triggers the box (not just mentioning "crises")
URGENT_IN_ANSWER = r"\b(fevers?|high temperature|temperature (of|above)|chest pain|breathing (difficult\w*|problems?)|difficulty breathing|priapism|painful erection)\b"

# Emergency rules work by CONCEPTS, not exact phrases, so different wordings are caught.
# A rule fires when every one of its word groups appears somewhere in the message.
EMERGENCY_RULES = {
    "breathing": [r"breath|breathe|breathing|breathless|wheez",
                  r"struggl|can'?t|cannot|difficult|hard|short|trouble|problem|tight|less"],
    "blue_lips": [r"(lips?|face|skin) (look|looks|are|is|turning|gone|going)? ?(blue|grey|gray)|blue lips"],
    "chest": [r"chest", r"pain|tight|hurt|pressure|heavy|ache|sore"],
    "fever": [r"fever|temperature|\btemp\b|shiver|feverish|hot and cold|feels? (very |really )?hot|burning up"],
    "stroke": [r"weak|numb|droop|slurr|confus|seizure|\bfit\b|fitting|unconscious|"
               r"passed out|pass out|faint|can'?t (speak|talk|move|walk|feel)|face (has )?dropp"],
    "headache": [r"headache|head ache|head hurts|head is hurting|head pain",
                 r"severe|worst|sudden|bad|terrible|awful|blur|vision|see|unbearable"],
    "vision": [r"blurr|blurry|vision|can'?t see|cannot see|losing (my )?sight|double vision"],
    "priapism": [r"erection|priapism|penis|private part"],
    "child_unwell": [r"floppy|drowsy|unresponsive|lethargic|won'?t wake|hard to wake|"
                     r"very pale|really pale|gone pale|swollen (tummy|belly|stomach)|"
                     r"(tummy|belly|stomach) is (swollen|big|hard)"],
    "agony": [r"\bagony\b|excruciating|unbearable"],
    "pain_worsening": [r"pain|hurt|crisis",
                       r"worse and worse|getting worse|keeps getting worse|nothing (is )?(working|helping)"],
    "severe_pain": [r"pain|hurt|crisis|ache",
                    r"unbearable|severe|worst|excruciating|agony|can'?t cope|can'?t take|"
                    r"10/10|ten out of ten|really bad|extreme|screaming"],
    "pain_relief_failing": [r"painkiller|pain ?killer|pain relief|medication|meds|medicine|codeine|"
                            r"morphine|tablets|paracetamol|ibuprofen|tramadol",
                            r"not (working|helping|touching)|isn'?t (working|helping|touching)|"
                            r"aren'?t (working|helping)|doesn'?t (work|help)|don'?t (work|help)|"
                            r"no (help|effect|difference)|stopped working|nothing (is )?(working|helping)|"
                            r"(isn'?t|not|doesn'?t) (doing|do) anything"],
    "vomiting": [r"vomit|being sick|throwing up|keep being sick|can'?t keep (anything|fluids|water|food) down"],
    "crisis_now": [r"crisis", r"start\w*|having|\bin (a )?crisis|begun|began|coming on|right now|\bnow\b"],
}

PERSONAL = (" i ", " i'm", " im ", " my ", " me ", " he ", " she ", " his ", " her ",
            " son", " daughter", " child", " baby", " kid", " husband", " wife", " partner")
EDUCATIONAL_STARTS = ("what is", "what are", "what's", "what causes", "what happens", "why",
                      "how", "explain", "tell me about", "define")
DOSE_PATTERNS = (
    r"how (much|many)\b.*\b(take|give|have|need|use)\b", r"\bdos(e|es|age|ing)\b",
    r"\bmax(imum)? (amount|daily)\b", r"\bhow often can (i|he|she|we)\b",
)
DOSE_IN_ANSWER = r"\b\d+(\.\d+)?\s?(mg|mcg|micrograms?|ml|g|grams?|tablets?|capsules?)\b|mg/kg"


def emergency_rule(q):
    """Return the name of the first emergency rule that matches, or None."""
    text = q.lower().replace("\u2019", "'")
    for name, groups in EMERGENCY_RULES.items():
        if all(re.search(g, text) for g in groups):
            return name
    return None


def is_emergency(q):
    text = f" {q.lower()} "
    if not emergency_rule(q):
        return False
    educational = text.strip().startswith(EDUCATIONAL_STARTS) and not any(p in text for p in PERSONAL)
    return not educational  # when in doubt, treat it as an emergency


# Second safety layer: a narrow yes/no question to the AI. Catches wordings the rules miss.
TRIAGE_PROMPT = """You are a safety filter for a sickle cell disease education app.
Decide if the message describes someone (the writer or a person they look after) who may
CURRENTLY have symptoms that could need urgent medical help. Examples: breathing difficulty,
chest pain, a high temperature, severe or uncontrolled pain, a crisis happening now, stroke
signs, a severe or sudden headache, changes in vision, a painful erection that won't go away,
repeated vomiting, a child who is drowsy, floppy, very pale or has a swollen tummy.
General questions asking for information (for example "What is acute chest syndrome?") are NO.
If you are unsure, answer YES.
Reply with only one word: YES or NO."""


def llm_says_emergency(question, llm):
    try:
        reply = llm.chat.completions.create(
            model=LLM_MODEL, temperature=0, max_tokens=300,
            messages=[{"role": "system", "content": TRIAGE_PROMPT},
                      {"role": "user", "content": question}],
        )
        verdict = (reply.choices[0].message.content or "").strip().upper()
        return verdict.startswith("YES")
    except Exception:
        return False   # if this check fails, the word rules still protect the user


def reply_type(text):
    if text == EMERGENCY_MESSAGE:
        return "emergency"
    if text == DOSE_MESSAGE:
        return "dose_refusal"
    if text == DONT_KNOW_MESSAGE:
        return "dont_know"
    return "answer"


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


# Spell out abbreviations and add synonyms so the search can find the right pages
EXPANSIONS = {
    r"\bSCD\b": "sickle cell disease", r"\bSCT\b": "sickle cell trait",
    r"\bSCA\b": "sickle cell anaemia", r"\bACS\b": "acute chest syndrome",
    r"\bVOC\b": "vaso-occlusive crisis pain crisis", r"\bHU\b": "hydroxycarbamide",
    r"\bhydroxy(carbamide|urea)\b": "hydroxycarbamide (hydroxyurea)",
    r"\bAS\b": "sickle cell trait carrier AS", r"\bSS\b": "sickle cell anaemia HbSS",
    r"\bgenotype\b": "genotype carrier inheritance",
}


def expand(question):
    for pattern, replacement in EXPANSIONS.items():
        question = re.sub(pattern, replacement, question)
    return question


def retrieve(question, embedder, collection):
    q_emb = embedder.encode([QUERY_PREFIX + expand(question)], normalize_embeddings=True).tolist()
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


# Phrases that reveal the behind-the-scenes setup. Only sentences containing these are removed.
LEAK = re.compile(
    r"(sources?|documents?|passages?|information) (you('ve| have) )?provided|you('ve| have) provided|"
    r"none of the (sources|documents|passages)|the (sources|documents|passages) (do not|don't|did not)|"
    r"based on the (sources|documents|passages)|according to the (sources|documents|passages)",
    re.IGNORECASE,
)
CITE_REMINDER = (
    "Your answer had no citations. Rewrite it so every sentence that states a fact ends with "
    "its citation, like [1]. Leave out anything you cannot cite. Do not mention sources, "
    "documents or what was provided."
)


def remove_leaks(text):
    out = []
    for line in text.splitlines():
        sentences = re.split(r"(?<=[.!?])\s+", line)
        kept = [x for x in sentences if not LEAK.search(x)]
        if kept or not line.strip():
            out.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def cited_only(text, chunks):
    """Keep only sources the answer actually cites, one per document, numbered 1, 2, 3..."""
    order, mapping = [], {}
    for n in re.findall(r"\[(\d+)\]", text):
        n = int(n)
        if n in mapping or not 1 <= n <= len(chunks):
            continue
        url = chunks[n - 1]["meta"]["url"]
        same = next((i for i, c in enumerate(order, 1) if c["meta"]["url"] == url), None)
        if same:
            mapping[n] = same
        else:
            order.append(chunks[n - 1])
            mapping[n] = len(order)

    def renumber(m):
        n = int(m.group(1))
        return f"[{mapping[n]}]" if n in mapping else ""
    text = re.sub(r"\[(\d+)\]", renumber, text)
    text = re.sub(r"(\[\d+\])(\1)+", r"\1", text)   # [1][1] becomes [1]
    return text, order


def ask_llm(llm, messages):
    reply = llm.chat.completions.create(model=LLM_MODEL, temperature=0.2, max_tokens=1500,
                                        messages=messages)
    return (reply.choices[0].message.content or "").strip()


def answer(question, embedder, collection, llm):
    # Layer 1: word rules (instant). Then doses. Layer 2: AI yes/no emergency check.
    if is_emergency(question):
        return EMERGENCY_MESSAGE, []
    if asks_for_dose(question):
        return DOSE_MESSAGE, []
    if llm_says_emergency(question, llm):
        return EMERGENCY_MESSAGE, []

    chunks = retrieve(question, embedder, collection)
    if not chunks:
        return DONT_KNOW_MESSAGE, []

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks)}]
    text = ask_llm(llm, messages)

    # No citations? Ask once more. Still none? Say "I don't know" rather than show uncited claims.
    if text and not re.search(r"\[\d+\]", text):
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": CITE_REMINDER}]
        text = ask_llm(llm, messages)
    if not re.search(r"\[\d+\]", text):
        return DONT_KNOW_MESSAGE, []

    text = remove_leaks(text)

    if re.search(DOSE_IN_ANSWER, text, flags=re.IGNORECASE):
        return DOSE_MESSAGE, []       # final safety net: never show a dose
    if not text:
        return DONT_KNOW_MESSAGE, []
    text, used = cited_only(text, chunks)
    if re.search(URGENT_TOPICS, question, flags=re.I) or re.search(URGENT_IN_ANSWER, text, flags=re.I):
        text += "\n\n" + URGENT_NOTE   # safety: always add the fixed NHS urgent advice
    return text, used


def format_sources(chunks):
    lines = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        lines.append(f"[{i}] {m['title']} | {m['owner']} | {m['licence']} | match {c['score']:.2f}\n    {m['url']}")
    return "\n".join(lines)


def load_everything():
    load_dotenv()
    embedder = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path="data/chroma").get_collection("sicklewise")
    llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
                 max_retries=5)   # wait and retry if the free tier says "too many requests"
    return embedder, collection, llm


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is acute chest syndrome?"
    embedder, collection, llm = load_everything()

    text, chunks = answer(question, embedder, collection, llm)
    print(f"\nQUESTION: {question}\n\n{text}\n")
    if chunks:
        print("SOURCES\n" + format_sources(chunks))
