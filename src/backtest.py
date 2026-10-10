"""Module A backtest: sentiment-driven weekly rebalancing vs equal weight.

    python src/backtest.py

Uses data/signals_cache.json (from ingest.py build) + weekly prices (yfinance, cached to data/prices.csv).
No look-ahead: weights chosen at Friday close of week W use only tweets up to that Friday and
earn the return of week W+1. Costs are ignored; turnover is reported instead.
"""
import json
import numpy as np, pandas as pd
import engine

K, DECAY, W_MIN, W_MAX = 0.8, 0.7, 0.2, 2.2  # tilt strength, weekly decay, weight floor/cap as multiples of 1/N
CACHE, PRICES = engine.ROOT / "data" / "signals_cache.json", engine.ROOT / "data" / "prices.csv"
DOCS = engine.ROOT / "docs"


def load_signals():
    s = [x for x in json.loads(CACHE.read_text()) if x["t"] < 99 and x.get("company")]  # historical tweets only
    df = pd.DataFrame(s)
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601").dt.tz_convert(None)
    df["week"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()  # week ends Friday
    df["score"] = df.sentiment_score * df.impact_score / 10  # signed, impact-weighted
    return df


def load_prices(tickers, start, end):
    if PRICES.exists():
        return pd.read_csv(PRICES, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.to_csv(PRICES)
    return px


def cap(w):
    n = len(w)
    for _ in range(30):  # iterative clip + renormalise
        w = w.clip(W_MIN / n, W_MAX / n); w = w / w.sum()
    return w


def run():
    df = load_signals()
    tickers = sorted(df.company.unique())
    px = load_prices(tickers, df.date.min() - pd.Timedelta(days=10), df.date.max() + pd.Timedelta(days=10))
    px = px.reindex(columns=tickers).resample("W-FRI").last().ffill()
    rets = px.pct_change().fillna(0)
    sig = df.pivot_table(index="week", columns="company", values="score", aggfunc="sum").reindex(columns=tickers)
    sig = sig.reindex(px.index).fillna(0)

    mood, prev = pd.Series(0.0, index=tickers), pd.Series(1 / len(tickers), index=tickers)
    rows, weights = [], {}
    dates = list(px.index)
    for d, nxt in zip(dates[:-1], dates[1:]):
        mood = DECAY * mood + sig.loc[d]
        w = cap(np.exp(K * mood) / np.exp(K * mood).sum())
        r_s = (w * rets.loc[nxt]).sum()
        r_e = rets.loc[nxt].mean()
        rows.append({"date": nxt, "sentiment": r_s, "equal_weight": r_e, "turnover": (w - prev).abs().sum() / 2})
        weights[d] = w; prev = w
    res = pd.DataFrame(rows).set_index("date")
    pd.DataFrame(weights).T.to_csv(DOCS / "backtest_weights.csv")

    def stats(r):
        eq = (1 + r).cumprod()
        return {"total_return": round(eq.iloc[-1] - 1, 4), "ann_vol": round(r.std() * 52 ** .5, 4),
                "sharpe": round(r.mean() / r.std() * 52 ** .5, 3) if r.std() else None,
                "max_drawdown": round((eq / eq.cummax() - 1).min(), 4)}
    out = {"weeks": len(res), "stocks": len(tickers), "tweets_used": len(df),
           "sentiment_strategy": {**stats(res.sentiment), "avg_weekly_turnover": round(res.turnover.mean(), 4)},
           "equal_weight": stats(res.equal_weight)}
    (DOCS / "backtest_results.json").write_text(json.dumps(out, indent=2))
    (1 + res[["sentiment", "equal_weight"]]).cumprod().to_csv(DOCS / "backtest_equity.csv")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ax = (1 + res[["sentiment", "equal_weight"]]).cumprod().plot(figsize=(8, 4), title="Growth of 1: sentiment vs equal weight")
        ax.set_ylabel("portfolio value"); plt.tight_layout(); plt.savefig(DOCS / "backtest_equity.png", dpi=150)
    except ImportError:
        print("pip install matplotlib to also save the chart")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    run()
