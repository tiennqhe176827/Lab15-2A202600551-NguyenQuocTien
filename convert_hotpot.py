"""Convert hotpot_dev_distractor_v1.json to QAExample format, pick 50 samples."""
import json
import random

random.seed(42)

raw = json.load(open("data/hotpot_dev_distractor_v1.json", "r", encoding="utf-8"))

context_lookup = {}
for title, sentences in raw[0]["context"]:
    context_lookup[title] = " ".join(sentences)
# Build lookup from all examples
for item in raw:
    for title, sentences in item["context"]:
        context_lookup[title] = " ".join(sentences)

difficulty_map = {"easy": "easy", "medium": "medium", "hard": "hard"}

examples = []
for item in raw:
    qid = item["_id"]
    question = item["question"]
    gold_answer = item["answer"]
    difficulty = difficulty_map.get(item.get("level", "medium"), "medium")

    sf = item.get("supporting_facts", [])
    if len(sf) < 2:
        # fallback: take first two context entries
        ctx_pairs = item["context"][:2]
    else:
        ctx_pairs = [sf[0], sf[1]]

    context_chunks = []
    for title, _ in ctx_pairs:
        text = context_lookup.get(title, "")
        if text:
            context_chunks.append({"title": title, "text": text})

    if len(context_chunks) < 2:
        # ensure at least 2 context chunks
        for t, s in item["context"][len(context_chunks):2]:
            context_chunks.append({"title": t, "text": " ".join(s)})

    examples.append({
        "qid": qid,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": gold_answer,
        "context": context_chunks[:2]
    })

random.shuffle(examples)
selected = examples[:50]

with open("data/hotpot_50.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print(f"Converted {len(selected)} examples -> data/hotpot_50.json")
