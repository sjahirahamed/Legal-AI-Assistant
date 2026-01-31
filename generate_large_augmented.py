# generate_large_augmented.py
import json
import random
from pathlib import Path

random.seed(42)

BASE = Path("data/test_cases.json")
OUT = Path("data/test_cases_big.json")

with open(BASE, "r", encoding="utf-8") as f:
    base = json.load(f)

TEMPLATES = [
    "{q}",
    "If this happened: {q}",
    "Scenario: {q}",
    "{q} Which IPC sections apply?",
    "{q} - identify the relevant IPC sections.",
    "User reports: {q}",
    "Short: {q}",
    "Report: {q}",
    "Please advise: {q}",
    "Legal query: {q}"
]

REPLACE_PAIRS = [
    ("man", ["person", "male", "individual"]),
    ("woman", ["person", "female", "lady"]),
    ("stole", ["took", "removed", "snatched"]),
    ("hit", ["struck", "collided with", "rammed"]),
    ("car", ["vehicle", "auto", "motor vehicle"]),
    ("knife", ["blade", "sharp instrument"]),
    ("killed", ["caused death", "murdered"]),
    ("threatened", ["intimidated", "menaced"]),
    ("cheated", ["defrauded", "scammed"])
]

def apply_replacements(q):
    s = q
    for orig, subs in REPLACE_PAIRS:
        if orig in s and random.random() < 0.5:
            s = s.replace(orig, random.choice(subs), 1)
    return s

def add_noise(q):
    if random.random() < 0.2:
        q = q.replace(",", "")
    if random.random() < 0.2:
        q = q + " Please help."
    return q

out = []
for case in base:
    q = case["query"]
    exp = case.get("expected_sections", [])
    out.append({"query": q, "expected_sections": exp})
    variants_per_case = 40
    for _ in range(variants_per_case):
        t = random.choice(TEMPLATES)
        q2 = t.format(q=q)
        if random.random() < 0.6:
            q2 = apply_replacements(q2)
        q2 = add_noise(q2)
        out.append({"query": q2, "expected_sections": exp})

random.shuffle(out)
MAX = 2000
out = out[:MAX]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(out)} cases to {OUT}")
