"""FinBERT-only sentiment over a larger, week-balanced tweet sample (no slow zero-shot step).

    python src/fast_signals.py --csv D:\\stock_tweets.csv --per-cell 15

Takes up to `per-cell` tweets per (stock, week). Tweets are processed in random order, so stopping early
(Ctrl+C) still leaves data spread across all weeks; re-running resumes. Output: data/sentiment_large.csv
"""
import argparse, time
import pandas as pd
import engine

OUT = engine.ROOT / "data" / "sentiment_large.csv"


def main(csv_path, per_cell, chunk=256):
    if engine.sentiment_backend() != "finbert":
        raise SystemExit("FinBERT did not load - refusing to write lexicon scores. Check the error above.")
    df = pd.read_csv(csv_path).rename(columns={"Date": "date", "Tweet": "text", "Stock Name": "company"})
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.text.str.len() >= 20].drop_duplicates("text")
    df["week"] = df["date"].dt.tz_convert(None).dt.to_period("W-FRI")
    df = df.sample(frac=1, random_state=42).groupby(["company", "week"]).head(per_cell)  # random order kept
    done = set(pd.read_csv(OUT)["id"]) if OUT.exists() else set()
    todo = df[~df.index.isin(done)]
    print(f"{len(df)} tweets selected, {len(done)} already done, {len(todo)} to go")
    t0, n0 = time.time(), 0
    for k in range(0, len(todo), chunk):
        part = todo.iloc[k:k + chunk]
        out = pd.DataFrame({"id": part.index, "company": part.company.values, "date": part.date.astype(str).values,
                            "sentiment": engine.sentiment_many(part.text.tolist())})
        out.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        n0 += len(part)
        rate = n0 / (time.time() - t0)
        print(f"  {k + len(part)}/{len(todo)} done ({rate:.1f} tweets/s, ~{(len(todo) - k - len(part)) / rate / 60:.0f} min left)")
    print("saved", OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True); p.add_argument("--per-cell", type=int, default=15)
    a = p.parse_args()
    main(a.csv, a.per_cell)
