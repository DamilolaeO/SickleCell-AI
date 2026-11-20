"""Turn every file in data/raw/ into searchable chunks stored in ChromaDB.

Steps: read each file -> split into chunks -> turn chunks into numbers
(embeddings) -> save them in a local database at data/chroma/.
Rebuilds from scratch each time, so rerun it whenever you add sources.
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

RAW = Path("data/raw")
DB_PATH = "data/chroma"
COLLECTION = "sicklewise"
MODEL = "BAAI/bge-small-en-v1.5"   # small, free embedding model that runs on a laptop
CHUNK_CHARS = 1000                  # roughly 150 to 200 words per chunk
OVERLAP_PARAS = 1                   # repeat the last paragraph so ideas are not cut in half
OUTDATED = ("voxelotor", "oxbryta")  # withdrawn 2024: never let the bot see these chunks

def read_file(path):
    """Split a file into its header (metadata) and its body text."""
    header, _, body = path.read_text(encoding="utf-8").partition("\n\n")
    meta = {}
    for line in header.splitlines():
        key, _, value = line.partition(": ")
        meta[key.strip().lower()] = value.strip()
    meta["source_id"] = path.stem
    meta["kind"] = "research" if path.stem.startswith("paper-") else "guidance"
    return meta, body


def chunk(text):
    """Group paragraphs into chunks of about CHUNK_CHARS characters."""
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    chunks, current = [], []
    for p in paras:
        if current and sum(len(x) for x in current) + len(p) > CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = current[-OVERLAP_PARAS:]
        current.append(p)
    if current:
        chunks.append("\n".join(current))
    return chunks


print("Loading embedding model (first run downloads about 130 MB)...")
model = SentenceTransformer(MODEL)

client = chromadb.PersistentClient(path=DB_PATH)
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass
collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

total = 0
for path in sorted(RAW.glob("*.txt")):
    meta, body = read_file(path)
    pieces = chunk(body)
    pieces = [p for p in pieces if not any(w in p.lower() for w in OUTDATED)]
    if not pieces:
        print(f"Skipped {path.name} (no text)")
        continue
    embeddings = model.encode(pieces, normalize_embeddings=True).tolist()
    collection.add(
        ids=[f"{meta['source_id']}-{i}" for i in range(len(pieces))],
        documents=pieces,
        embeddings=embeddings,
        metadatas=[{
            "source_id": meta["source_id"],
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "owner": meta.get("owner", ""),
            "licence": meta.get("licence", ""),
            "population": meta.get("population", ""),
            "kind": meta["kind"],
        } for _ in pieces],
    )
    total += len(pieces)
    print(f"{path.name}: {len(pieces)} chunks")

print(f"\nDone. {total} chunks saved to {DB_PATH}")
