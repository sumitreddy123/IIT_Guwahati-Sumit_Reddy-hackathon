"""Data ingestion + caching, so the demo never needs live internet or a long model run.

  python src/ingest.py sample --csv D:\\stock_tweets.csv --n 1500   # -> data/tweets_sample.csv (commit it)
  python src/ingest.py news                                        # -> data/news_cache.json (needs NEWSAPI_KEY)
  python src/ingest.py build                                       # -> data/signals_cache.json (runs the NLP engine)

`build` is resumable: items already in the cache are skipped, so you can stop and re-run it.
"""
import argparse, json, math, os
import pandas as pd
import engine

DATA = engine.ROOT / "data"
TWEETS, NEWS, CACHE = DATA / "tweets_sample.csv", DATA / "news_cache.json", DATA / "signals_cache.json"
QUERIES = {"TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "META": "Meta Platforms",
           "NFLX": "Netflix", "AMD": "AMD", "BA": "Boeing", "DIS": "Disney", "KO": "Coca-Cola",
           "PYPL": "PayPal", "INTC": "Intel"}


def sample_tweets(csv, n):
    """Stratified sample (equal tweets per stock) + weekly time step `t` computed on the full date range."""
    df = pd.read_csv(csv).rename(columns={"Date": "date", "Tweet": "text", "Stock Name": "company", "Company Name": "name"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.text.str.len() >= 20].drop_duplicates("text")
    df["t"] = (df["date"] - df["date"].min()).dt.days // 7
    per = math.ceil(n / df.company.nunique())
    out = df.sample(frac=1, random_state=42).groupby("company").head(per)
    out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(TWEETS, index=False)
    print(f"{len(df)} usable tweets, {df.company.nunique()} stocks, {df.date.min().date()} to {df.date.max().date()}")
    print(f"saved {len(out)} tweets -> {TWEETS} ({TWEETS.stat().st_size // 1024} kB)")


def fetch_news(per=20):
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        raise SystemExit("Set NEWSAPI_KEY first (Windows cmd: set NEWSAPI_KEY=yourkey)")
    import requests
    items, seen = [], set()
    for tkr, q in QUERIES.items():
        r = requests.get("https://newsapi.org/v2/everything", timeout=15, params={
            "q": f'"{q}" stock', "pageSize": per, "language": "en", "sortBy": "publishedAt", "apiKey": key}).json()
        if r.get("status") != "ok":
            print(tkr, "failed:", r.get("message")); continue
        for a in r["articles"]:
            if a["url"] in seen or not a.get("title"):
                continue
            seen.add(a["url"])
            items.append({"id": f"news{len(items)}", "t": 99, "source": "news", "company": tkr,
                          "date": a["publishedAt"], "text": f"{a['title']}. {a.get('description') or ''}".strip()})
        print(tkr, "->", len(items), "articles so far")
    NEWS.write_text(json.dumps(items, indent=1)); print("saved", NEWS)


def load_items():
    items = []
    if TWEETS.exists():
        for i, r in pd.read_csv(TWEETS).iterrows():
            items.append({"id": f"tw{i}", "t": int(r.t), "source": "twitter", "company": r.company,
                          "date": str(r.date), "text": r.text})
    if NEWS.exists():
        items += json.loads(NEWS.read_text())
    return items


def build(chunk=50):
    cache = {s["id"]: s for s in json.loads(CACHE.read_text())} if CACHE.exists() else {}
    todo = [i for i in load_items() if i["id"] not in cache]
    print(f"{len(cache)} cached, {len(todo)} to analyze (backend: {engine.sentiment_backend()})")
    for k in range(0, len(todo), chunk):
        part = todo[k:k + chunk]
        for item, s in zip(part, engine.sentiment_many([i["text"] for i in part])):
            cache[item["id"]] = engine.analyze(item, s)
        CACHE.write_text(json.dumps(list(cache.values()), indent=1))  # save progress
        print(f"  {len(cache)} done")
    print("saved", CACHE)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sample", "news", "build"])
    p.add_argument("--csv"); p.add_argument("--n", type=int, default=1500)
    a = p.parse_args()
    if a.cmd == "sample":
        sample_tweets(a.csv, a.n)
    elif a.cmd == "news":
        fetch_news()
    else:
        build()
