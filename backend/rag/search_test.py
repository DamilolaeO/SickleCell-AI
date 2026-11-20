"""Ask the library a question and see which chunks it finds. No AI model involved."""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # this model works best with it

question = " ".join(sys.argv[1:]) or "Why does cold weather trigger a crisis?"

model = SentenceTransformer(MODEL)
collection = chromadb.PersistentClient(path="data/chroma").get_collection("sicklewise")

q_emb = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()
results = collection.query(query_embeddings=q_emb, n_results=4)

print(f"\nQUESTION: {question}\n")
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"--- {meta['title']} ({meta['owner']}) | match: {1 - dist:.2f}")
    print(f"    {meta['url']}")
    print("    " + doc[:300].replace("\n", " ") + "...\n")
