"""Evaluate sentiment on FinancialPhraseBank (data/all-data.csv). Run once per backend:
    NLP_BACKEND=lexicon python src/evaluate_sentiment.py   # baseline
    python src/evaluate_sentiment.py                       # FinBERT (if available)"""
import json, sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import engine

T = 0.2  # |score| above this counts as positive/negative
df = pd.read_csv(engine.ROOT / "data" / "all-data.csv", header=None, names=["label", "text"], encoding="latin-1")
backend = engine.sentiment_backend()
pred = ["positive" if s > T else "negative" if s < -T else "neutral" for s in engine.sentiment_many(df.text.tolist())]
res = {"backend": backend, "n": len(df), "accuracy": round(accuracy_score(df.label, pred), 3),
       "macro_f1": round(f1_score(df.label, pred, average="macro"), 3)}
print(res); print(confusion_matrix(df.label, pred, labels=["negative", "neutral", "positive"]))
out = engine.ROOT / "docs" / f"sentiment_eval_{backend}.json"; out.write_text(json.dumps(res, indent=2))
print("saved", out)
