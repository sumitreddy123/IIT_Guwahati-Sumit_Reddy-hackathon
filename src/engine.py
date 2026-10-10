"""AI/NLP Risk Engine: text -> sentiment, event class, tickers, impact score.
Backends: FinBERT + zero-shot BART (if transformers installed and models downloadable),
otherwise a lexicon/keyword fallback. Force fallback with NLP_BACKEND=lexicon."""
import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE, OUT = ROOT / "data" / "sample_texts.json", ROOT / "outputs" / "signals.json"
MODE = os.getenv("NLP_BACKEND", "auto")  # auto | lexicon

EVENTS = {  # keywords, base severity (1-10)
    "Geopolitical": ("war sanction missile conflict invasion tension tariff embargo ceasefire".split(), 7),
    "Macroeconomic": ("inflation fed rate gdp unemployment recession cpi slowdown".split(), 6),
    "Credit Event": ("default downgrade bankruptcy debt insolvency restructuring".split(), 8),
    "Merger/Acquisition": ("acquire acquisition merger takeover buyout deal".split(), 5),
    "Product Launch": ("launch unveil launches unveils release introduces".split(), 3),
}
POS = set("strong record beats upbeat bullish gains growth lift surge easing ease stabilize stabilizes progress agrees fade".split())
NEG = set("fall worried plunges crash downgraded default bankruptcy lawsuit fine fears escalate strikes recall sell-off risk slowdown brace".split())
INTENSE = set("sharply massive crash plunges record surge escalate".split())
SRC_W = {"news": 1.0, "twitter": 0.8}
ALIASES = {"apple": "AAPL", "microsoft": "MSFT", "jpmorgan": "JPM", "exxon": "XOM", "tesla": "TSLA",
           "nvidia": "NVDA", "amazon": "AMZN", "meta": "META", "facebook": "META",
           "boeing": "BA", "google": "GOOGL", "alphabet": "GOOGL"}

_pipes = {}
def _pipe(task, model):
    """Load a HF pipeline once; None if unavailable (offline, not installed, lexicon mode)."""
    if MODE == "lexicon":
        return None
    if task not in _pipes:
        try:
            from transformers import pipeline
            _pipes[task] = pipeline(task, model=model, **({"top_k": None} if task == "text-classification" else {}))
        except Exception as e:
            print(f"[engine] {model} unavailable ({type(e).__name__}); using fallback")
            _pipes[task] = None
    return _pipes[task]

FINBERT = ("text-classification", "ProsusAI/finbert")
ZEROSHOT = ("zero-shot-classification", "facebook/bart-large-mnli")

def sentiment_backend():
    return "finbert" if _pipe(*FINBERT) else "lexicon"

def _lex(text):
    w = [x.strip(".,$").lower() for x in text.split()]
    p, n = sum(x in POS for x in w), sum(x in NEG for x in w)
    return round((p - n) / (p + n), 3) if p + n else 0.0

def sentiment_many(texts):
    """Scores in [-1, 1] = P(positive) - P(negative). Batched for speed."""
    p = _pipe(*FINBERT)
    if not p:
        return [_lex(t) for t in texts]
    out = p([t[:512] for t in texts], batch_size=32, truncation=True)  # list of lists of dicts
    return [round({d["label"].lower(): d["score"] for d in r}.get("positive", 0)
                  - {d["label"].lower(): d["score"] for d in r}.get("negative", 0), 3) for r in out]

def sentiment(text):
    return sentiment_many([text])[0]

def classify_event(text):
    """Returns (label, confidence). Zero-shot if available, else keyword match."""
    z = _pipe(*ZEROSHOT)
    if z:
        r = z(text, candidate_labels=list(EVENTS) + ["Other"])
        return r["labels"][0], round(r["scores"][0], 2)
    low = text.lower()
    scores = {e: sum(k in low for k in kw) for e, (kw, _) in EVENTS.items()}
    best = max(scores, key=scores.get)
    return (best, 0.6) if scores[best] else ("Other", 0.3)

def extract_tickers(text):
    found = re.findall(r"\$([A-Z]{1,5})\b", text)
    low = text.lower()
    found += [t for name, t in ALIASES.items() if re.search(rf"\b{name}\b", low)]
    return list(dict.fromkeys(found))

def impact(text, event, sent, source):
    base = EVENTS.get(event, (None, 2))[1]
    boost = 1 + 0.15 * sum(w.strip(".,").lower() in INTENSE for w in text.split())
    return round(min(10, max(1, base * (0.5 + abs(sent)) * SRC_W.get(source, 0.8) * boost)), 1)

def analyze(item, sent=None):
    s = sent if sent is not None else sentiment(item["text"])
    e, conf = classify_event(item["text"])
    tickers = extract_tickers(item["text"])
    return {**item, "company": item.get("company") or (tickers[0] if tickers else None),
            "tickers": tickers, "sentiment_score": s, "event_class": e, "event_confidence": conf,
            "impact_score": impact(item["text"], e, s, item["source"])}

def fetch_newsapi(query="stock market", n=20):
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []
    import requests
    r = requests.get("https://newsapi.org/v2/everything",
                     params={"q": query, "pageSize": n, "language": "en", "apiKey": key}, timeout=10)
    return [{"id": f"api{i}", "t": 99, "source": "news", "company": None,
             "text": f"{a['title']}. {a.get('description') or ''}"} for i, a in enumerate(r.json().get("articles", []))]

def ingest():
    return json.loads(SAMPLE.read_text()) + fetch_newsapi()

def run(write=True):
    items = ingest()
    scores = sentiment_many([i["text"] for i in items])
    signals = [analyze(i, s) for i, s in zip(items, scores)]
    if write:
        OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(signals, indent=2))
    return signals

if __name__ == "__main__":
    print("sentiment backend:", sentiment_backend())
    for s in run():
        print(f"{str(s['company']):5} {s['source']:8} sent={s['sentiment_score']:+.2f} {s['event_class']:18} conf={s['event_confidence']} impact={s['impact_score']}")
