# train_query_classifier.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
import joblib
import warnings

# Silence a harmless sklearn future warning in many environments
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path("data/test_cases_big.json")
MODEL_OUT = Path("models")
MODEL_OUT.mkdir(exist_ok=True, parents=True)

print("Loading dataset...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

print("Loading law sections...")
with open("data/laws.json", "r", encoding="utf-8") as f:
    laws = json.load(f)
sections = [law["section"] for law in laws]

label_encoder = LabelEncoder()
label_encoder.fit(sections)

queries = []
labels = []
for c in cases:
    q = c["query"]
    exp = c.get("expected_sections", [])
    if not exp:
        continue
    primary = exp[0]
    if primary not in sections:
        continue
    queries.append(q)
    labels.append(primary)

print(f"Total training examples: {len(queries)}")

print("Loading MPNet model for embeddings...")
embed_model = SentenceTransformer("all-mpnet-base-v2")
batch_size = 64

def embed_texts(texts):
    return embed_model.encode(texts, show_progress_bar=True, batch_size=batch_size)

X = embed_texts(queries)
y = label_encoder.transform(labels)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.12, random_state=42, stratify=y
)

print("Training LogisticRegression classifier (single-process to avoid Windows joblib issues)...")
# IMPORTANT: set n_jobs=1 to avoid joblib multiprocessing / _posixsubprocess error on some Windows builds
clf = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs",
    C=1.0,
    n_jobs=1
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
top3 = top_k_accuracy_score(y_val, clf.predict_proba(X_val), k=3)
print(f"Validation accuracy (top1): {acc:.4f}")
print(f"Validation top-3 accuracy: {top3:.4f}")

# A readable classification report — use label names mapped via label_encoder
target_names = label_encoder.inverse_transform(sorted(set(y_val)))
print("\nClassification report (for validation set labels present):")
print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

joblib.dump({"clf": clf, "le": label_encoder}, MODEL_OUT / "query_classifier.joblib")
print(f"Saved classifier -> {MODEL_OUT / 'query_classifier.joblib'}")
