"""Module A: Tactical Index Rebalancing driven by sentiment."""
import numpy as np, pandas as pd

UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "TSLA", "NVDA", "AMZN", "META", "BA", "GOOGL"]
K, W_MIN, W_MAX = 0.8, 0.02, 0.20  # tilt strength, weight floor/cap


def prices(days=30, seed=7):
    """Real prices via yfinance if installed, else a seeded random walk."""
    try:
        import yfinance as yf
        df = yf.download(UNIVERSE, period=f"{days}d", progress=False)["Close"].dropna()
        if len(df) > 5:
            return df
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, len(UNIVERSE))), 0)), columns=UNIVERSE)


def _cap(w):
    for _ in range(20):  # iterative clip + renormalise
        w = (w.clip(W_MIN, W_MAX)) / w.clip(W_MIN, W_MAX).sum()
    return w


def rebalance(signals):
    """Cumulative sentiment per ticker (decayed per step) -> weights history."""
    w = pd.Series(1 / len(UNIVERSE), index=UNIVERSE)
    mood = pd.Series(0.0, index=UNIVERSE)
    hist = {}
    for t in sorted({s["t"] for s in signals if s["t"] < 99}):
        mood *= 0.7  # sentiment decays
        for s in signals:
            if s["t"] == t and s["company"] in mood.index:
                mood[s["company"]] += s["sentiment_score"] * s["impact_score"] / 10
        raw = (1 / len(UNIVERSE)) * np.exp(K * mood)
        w = _cap(raw / raw.sum())
        hist[f"step {t}"] = w
    return pd.DataFrame(hist).T
