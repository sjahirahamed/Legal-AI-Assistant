# generate_augmented_testcases.py
import json
import random
import copy
from pathlib import Path

# Load original test cases
base_path = Path("data")
src = base_path / "test_cases.json"
dst = base_path / "test_cases_augmented.json"

with open(src, "r", encoding="utf-8") as f:
    base = json.load(f)

# small paraphrase templates
TEMPLATES = [
    "{q}",
    "In one case: {q}",
    "{q} What sections apply?",
    "If the following happened: {q}",
    "{q} — which IPC sections would apply?",
    "Explain applicable laws for: {q}",
    "User reported: {q}",
    "{q} Please identify the relevant IPC sections."
]

# small rewording swaps to insert
REWORDS = [
    ("man", ["person", "male", "individual"]),
    ("woman", ["female", "lady", "individual"]),
    ("stole", ["took", "removed", "snatched"]),
    ("hit", ["struck", "collided with", "knocked"]),
    ("kill", ["cause death of", "murder", "slay"]),
    ("cheated", ["defrauded", "scammed", "deceived"]),
    ("threatened", ["intimidated", "menaced", "warned"]),
    ("car", ["vehicle", "auto", "motor vehicle"]),
    ("knife", ["blade", "knife"]),
    ("stabbing", ["stabbed", "cut with a knife"]),
    ("robbed", ["held up", "mugged", "robbed"])
]

# synonyms pool (small) — add more as needed
SYN_POOL = {
    "hurt": ["injured", "wounded"],
    "injuries": ["harm", "wounds"],
    "death": ["fatality", "fatal"],
    "steal": ["theft", "robbery", "snatch"],
    "threaten": ["intimidate", "menace"]
}

def apply_rewords(text):
    t = text
    for orig, subs in REWORDS:
        if orig in t:
            rep = random.choice(subs)
            t = t.replace(orig, rep, 1)
    return t

def insert_synonym_phrases(text):
    # append short synonyms randomly
    extras = []
    for k, v in SYN_POOL.items():
        if k in text and random.random() < 0.6:
            extras.append(random.choice(v))
    if extras:
        return text + " | " + " ".join(extras)
    return text

augmented = []
for case in base:
    q = case["query"]
    expected = case.get("expected_sections", [])
    # keep original
    augmented.append({"query": q, "expected_sections": expected})
    # produce variations
    for _ in range(9):  # 9 variations -> 10 per base case
        template = random.choice(TEMPLATES)
        q2 = template.format(q=q)
        # randomly reword
        if random.random() < 0.6:
            q2 = apply_rewords(q2)
        # random synonym append
        if random.random() < 0.5:
            q2 = insert_synonym_phrases(q2)
        # simple random punctuation paraphrase
        if random.random() < 0.3:
            q2 = q2.replace(",", "")  # remove commas sometimes
        augmented.append({"query": q2, "expected_sections": expected})

# optionally shuffle
random.shuffle(augmented)

# limit to ~500 (if you started with 50 -> 10x => 500)
max_cases = 500
augmented = augmented[:max_cases]

with open(dst, "w", encoding="utf-8") as f:
    json.dump(augmented, f, indent=2, ensure_ascii=False)

print(f"Generated {len(augmented)} augmented test cases -> {dst}")
