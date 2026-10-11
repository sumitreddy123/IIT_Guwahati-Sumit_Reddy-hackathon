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
    "Macroeconomic": ("inflation fed gdp unemployment recession cpi slowdown".split() + ["interest rate", "rate hike", "rate cut"], 6),
    "Credit Event": ("default downgrade bankruptcy debt insolvency restructuring".split(), 8),
    "Merger/Acquisition": ("acquire acquisition merger takeover buyout deal".split(), 5),
    "Product Launch": ("launch unveil release introduce".split(), 3),
    "Earnings": ("earnings beats profit revenue guidance expectations quarterly".split(), 4),
    "Regulatory/Legal": ("lawsuit fine regulation regulator regulatory legislation tax probe antitrust sec investigation taxation levy ban crackdown penalty penalties".split(), 6),
    "Operational/Product Incident": (["system crash", "system failure", "outage", "glitch", "malfunction", "defect", "recall",
                                      "data breach", "cyberattack", "hack", "safety"], 6),
    "Analyst Rating/Price Target": (["analyst", "price target", "upgrade", "initiates coverage", "overweight", "underweight", "outperform"], 3),
    "Management/Corporate Action": (["steps down", "resign", "new ceo", "layoff", "buyback", "dividend", "stock split", "spin-off",
                                     "share repurchase", "hiring freeze"], 5),
    "Market Commentary": ([], 2),  # opinion / price talk / technical analysis: no discrete event (zero-shot label only)
}
# zero-shot works better with descriptive labels than with bare class names
LABELS = {
    "geopolitical conflict, war or sanctions": "Geopolitical",
    "macroeconomic news such as inflation, interest rates or GDP": "Macroeconomic",
    "credit event such as a default, downgrade or bankruptcy": "Credit Event",
    "merger or acquisition": "Merger/Acquisition",
    "product launch": "Product Launch",
    "company earnings or financial performance": "Earnings",
    "government regulation, tax policy, lawsuit or legal investigation": "Regulatory/Legal",
    "operational failure, system outage, product defect, recall or safety incident": "Operational/Product Incident",
    "analyst rating, upgrade or price target change": "Analyst Rating/Price Target",
    "management change, layoffs, buyback, dividend or stock split": "Management/Corporate Action",
    "market commentary, opinion, price movement or technical analysis": "Market Commentary",
    "other": "Other",
}
POS = set("strong record beats upbeat bullish gains growth lift surge easing ease stabilize stabilizes progress agrees fade".split())
NEG = set("fall worried plunges crash downgraded default bankruptcy lawsuit fine fears escalate strikes recall sell-off risk slowdown brace".split())
INTENSE = set("sharply massive crash plunges record surge escalate".split())
SRC_W = {"news": 1.0, "twitter": 0.8}
ALIASES = {"apple": "AAPL", "microsoft": "MSFT", "jpmorgan": "JPM", "exxon": "XOM", "tesla": "TSLA",
           "nvidia": "NVDA", "amazon": "AMZN", "meta": "META", "facebook": "META",
           "boeing": "BA", "google": "GOOGL", "alphabet": "GOOGL"}

# hand-written sector -> tickers (approximate; edit to match your universe)
SECTORS = {"e-commerce": ["AMZN"], "ecommerce": ["AMZN"], "online retail": ["AMZN"], "social media": ["META"],
           "electric vehicle": ["TSLA", "F", "NIO", "XPEV"], "electric car": ["TSLA", "F", "NIO", "XPEV"],
           "automaker": ["TSLA", "F"], "streaming": ["NFLX", "DIS"], "semiconductor": ["AMD", "INTC", "TSM"],
           "chipmaker": ["AMD", "INTC", "TSM"], "cloud": ["MSFT", "AMZN", "CRM"], "airline": ["BA"],
           "aerospace": ["BA", "NOC"], "payments": ["PYPL"]}

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
    low = text.lower()
    hits = {e: sum(bool(re.search(rf"\b{k}(?:s|es|d|ed|ing)?\b", low)) for k in kw) for e, (kw, _) in EVENTS.items()}  # whole words only
    kbest = max(hits, key=hits.get)
    z = _pipe(*ZEROSHOT)
    if not z:
        return (kbest, 0.6) if hits[kbest] else ("Other", 0.3)
    r = z(text, candidate_labels=list(LABELS))
    label, conf = LABELS[r["labels"][0]], round(r["scores"][0], 2)
    if conf < 0.6 and hits[kbest]:  # weak model confidence: trust a keyword match
        return kbest, conf
    if conf < 0.4:
        return "Other", conf
    return label, conf

def extract_tickers(text):
    found = re.findall(r"\$([A-Z]{1,5})\b", text)
    low = text.lower()
    found += [t for name, t in ALIASES.items() if re.search(rf"\b{name}\b", low)]
    for phrase, ts in SECTORS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            found += ts
    return list(dict.fromkeys(found))

def impact(text, event, sent, source):
    base = EVENTS.get(event, (None, 2))[1]
    boost = 1 + 0.15 * sum(w.strip(".,").lower() in INTENSE for w in text.split())
    return round(min(10, max(1, base * (0.5 + abs(sent)) * SRC_W.get(source, 0.8) * boost)), 1)

# Rule layer (heuristic, NOT validated): FinBERT scores burden-without-negative-words, e.g. "twice the
# tax rates", as neutral. Applied only to Regulatory/Legal or Geopolitical events with a weak score and
# no earnings context; relief words ("tax cut") flip the sign. Raw model score is kept as sentiment_raw.
POLICY_NEG = set("tax taxes taxation tariff tariffs levy penalty penalties crackdown ban banned".split())
POLICY_POS = set("cut cuts cutting relief exemption exempt lower reduce reduced scrapped repeal".split())
EARNINGS_WORDS = set("profit profits revenue revenues earnings sales income quarter quarterly".split())

def policy_adjust(text, s, event):
    if event not in ("Regulatory/Legal", "Geopolitical") or abs(s) >= 0.25:
        return s, False
    w = {x.strip(".,$;:").lower() for x in text.split()}
    if w & EARNINGS_WORDS:
        return s, False
    neg, pos = len(w & POLICY_NEG), len(w & POLICY_POS)
    if not (neg or pos):
        return s, False
    s += min(0.6, 0.3 * pos) if pos else max(-0.6, -0.3 * neg)  # relief wins
    return round(max(-1, min(1, s)), 3), True

# Per-stock sentiment for texts naming several stocks. Text-level sentiment would move every named stock the same
# way, so a price-move word attaches to the nearest stock name (within 3 words; on a tie the name before it) and sets
# that stock's sign. Stocks with no such cue keep the text-level score. A heuristic, not true targeted sentiment.
DOWN = set("fall falls falling fell drop drops dropping dropped plunge plunges plunging plunged slump slumps slumped tumble tumbles tumbled sink sinks sinking sank slide slides sliding slid decline declines declining declined tank tanks tanked crash crashes crashed lose loses losing lost".split())
UP = set("rise rises rising rose surge surges surging surged jump jumps jumping jumped soar soars soaring soared rally rallies rallied gain gains gaining gained climb climbs climbing climbed".split())

def entity_sentiments(text, tickers, doc_score):
    if len(tickers) < 2:
        return {}
    toks = re.findall(r"[$\w'-]+", text.lower())
    names = {t: {n for n, tk in ALIASES.items() if tk == t} | {"$" + t.lower()} for t in tickers}
    pos = [(i, t) for i, w in enumerate(toks) for t in tickers if w in names[t]]
    votes = {t: 0 for t in tickers}
    for i, w in enumerate(toks):
        d = -1 if w in DOWN else 1 if w in UP else 0
        near = [(abs(i - j), 0 if j < i else 1, t) for j, t in pos if abs(i - j) <= 3] if d else []
        if near:
            votes[min(near)[2]] += d
    if not any(votes.values()):
        return {}
    return {t: (0.6 if v > 0 else -0.6 if v < 0 else doc_score) for t, v in votes.items()}

def analyze(item, sent=None):
    raw = sent if sent is not None else sentiment(item["text"])
    e, conf = classify_event(item["text"])
    s, adjusted = policy_adjust(item["text"], raw, e)
    tickers = extract_tickers(item["text"])
    return {**item, "company": item.get("company") or (tickers[0] if tickers else None),
            "tickers": tickers, "entity_sentiment": entity_sentiments(item["text"], tickers, s), "sentiment_score": s, "sentiment_raw": raw, "policy_adjusted": adjusted,
            "event_class": e, "event_confidence": conf, "impact_score": impact(item["text"], e, s, item["source"])}

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
