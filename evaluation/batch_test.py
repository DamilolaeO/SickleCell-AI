"""Run every question in evaluation/test_set.csv and score the results.

Run from the project folder:
    python evaluation/batch_test.py
Saves:
    evaluation/batch_results.md   full answers, for reading and review
    evaluation/batch_results.csv  one row per question with PASS or FAIL
"""
import csv, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "backend/app")
import ask  # noqa: E402

TEST_SET = Path("evaluation/test_set.csv")
OUT_MD = Path("evaluation/batch_results.md")
OUT_CSV = Path("evaluation/batch_results.csv")

# Works whether the file uses commas or tabs (copying from a preview can turn commas into tabs)
text = TEST_SET.read_text(encoding="utf-8-sig")
delim = "\t" if "\t" in text.splitlines()[0] else ","
rows = [{k.strip(" ,"): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(text.splitlines(), delimiter=delim)]
# Optional: test only some categories, e.g.  python evaluation/batch_test.py emergency dose
if len(sys.argv) > 1:
    rows = [r for r in rows if r["category"] in sys.argv[1:]]
print(f"Loading models... ({len(rows)} questions)")
embedder, collection, llm = ask.load_everything()

results, md = [], [f"# Sicklewise batch test, {datetime.now():%d %b %Y %H:%M}\n"]
for i, row in enumerate(rows, 1):
    q, expected = row["question"], row["expected"]
    try:
        text, sources = ask.answer(q, embedder, collection, llm)
        got = ask.reply_type(text)
    except Exception as e:
        text, sources, got = f"ERROR: {e}", [], "error"
    passed = got == expected
    results.append({**row, "got": got, "result": "PASS" if passed else "FAIL",
                    "citations": text.count("[") if got == "answer" else ""})
    print(f"[{i}/{len(rows)}] {'PASS' if passed else 'FAIL'}  {row['category']:<11} {q}")

    md.append(f"## {i}. {q}\n\n*Category:* {row['category']} | *Expected:* {expected} | "
              f"*Got:* {got} | **{'PASS' if passed else 'FAIL'}**\n\n{text}\n")
    for n, c in enumerate(sources, 1):
        m = c["meta"]
        md.append(f"{n}. {m['title']} ({m['kind']}) | {m['url']}")
    md.append("\n---\n")
    # Free Groq tier allows about 8,000 words-worth (tokens) per minute.
    # Full answers use a lot, so pause longer after them.
    time.sleep(20 if got == "answer" else 3)

# Summary by category
by_cat = defaultdict(lambda: [0, 0])
for r in results:
    by_cat[r["category"]][1] += 1
    by_cat[r["category"]][0] += r["result"] == "PASS"
summary = ["## Summary\n", "| Category | Passed |", "|---|---|"]
for cat, (p, n) in by_cat.items():
    summary.append(f"| {cat} | {p}/{n} |")
missed = [r["question"] for r in results if r["expected"] == "emergency" and r["got"] != "emergency"]
summary.append(f"\n**Missed emergencies: {len(missed)}** (target: 0)")
summary += [f"- {q}" for q in missed]
summary.append("")

OUT_MD.write_text("\n".join(md[:1] + summary + md[1:]), encoding="utf-8")
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

print("\n" + "\n".join(summary))
print(f"Saved {OUT_MD} and {OUT_CSV}")
