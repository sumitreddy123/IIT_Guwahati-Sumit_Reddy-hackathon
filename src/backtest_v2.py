"""Larger-sample test of the sentiment signal. Settings are fixed in advance (no tuning).

    python src/backtest_v2.py

Signal: weekly mean FinBERT sentiment per stock, z-scored across stocks.
Variants (declared before looking at results):
  A) long-only tilt, w ~ exp(K*z), vs equal weight
  B) long-short: long top quintile, short bottom quintile (dollar neutral)
Weights set at Friday close earn the NEXT week's return. Costs: COST per unit of traded notional.
Reported separately for the first half of weeks (train) and the second half (test), plus rank IC.
"""
import json
import numpy as np, pandas as pd
import backtest as b
import engine

K, COST = 0.3, 0.0010  # fixed in advance; COST = 10 bps per unit traded (an assumption)
LARGE, DOCS = engine.ROOT / "data" / "sentiment_large.csv", engine.ROOT / "docs"


def stats(r):
    eq = (1 + r).cumprod()
    return {"total_return": round(eq.iloc[-1] - 1, 4), "sharpe": round(r.mean() / r.std() * 52 ** .5, 3) if r.std() else None,
            "max_drawdown": round((eq / eq.cummax() - 1).min(), 4), "weeks": len(r)}


def run():
    s = pd.read_csv(LARGE)
    s["date"] = pd.to_datetime(s["date"], utc=True, format="ISO8601").dt.tz_convert(None)
    s["week"] = s["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    tickers = sorted(s.company.unique()); n = len(tickers)
    px = b.load_prices(tickers, s.date.min() - pd.Timedelta(days=10), s.date.max() + pd.Timedelta(days=10))
    px = px.reindex(columns=tickers).resample("W-FRI").last().ffill()
    rets = px.pct_change().fillna(0)
    sig = s.pivot_table(index="week", columns="company", values="sentiment", aggfunc="mean").reindex(index=px.index, columns=tickers)
    z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1), axis=0).fillna(0)  # 0 = no information

    eqw, k = pd.Series(1 / n, index=tickers), max(1, round(n * 0.2))
    prevA, prevB, rows, ics = eqw, pd.Series(0.0, index=tickers), [], []
    dates = list(px.index); wts = {}
    for d, nxt in zip(dates[:-1], dates[1:]):
        zz, r = z.loc[d], rets.loc[nxt]
        wA = eqw if (zz == 0).all() else b.cap(np.exp(K * zz) / np.exp(K * zz).sum())
        wB = pd.Series(0.0, index=tickers)
        if not (zz == 0).all():
            order = zz.sort_values().index; wB[order[-k:]], wB[order[:k]] = 1 / k, -1 / k
        ic = zz.corr(r, method="spearman") if zz.std() > 0 else np.nan
        rows.append({"date": nxt, "equal_weight": r.mean(),
                     "long_only_gross": (wA * r).sum(), "long_only_net": (wA * r).sum() - COST * (wA - prevA).abs().sum(),
                     "long_short_gross": (wB * r).sum(), "long_short_net": (wB * r).sum() - COST * (wB - prevB).abs().sum(), "ic": ic})
        prevA, prevB = wA, wB; wts[d] = wA
    res = pd.DataFrame(rows).set_index("date")
    mid = len(res) // 2
    out = {"stocks": n, "tweets_used": len(s), "tweets_per_stock_week": round(len(s) / (n * len(res)), 1), "K": K, "cost_bps": COST * 1e4}
    for name, part in {"all": res, "train_first_half": res.iloc[:mid], "test_second_half": res.iloc[mid:]}.items():
        ic = part.ic.dropna()
        out[name] = {"mean_rank_ic": round(ic.mean(), 4), "ic_t_stat": round(ic.mean() / ic.std() * len(ic) ** .5, 2) if len(ic) > 2 else None,
                     **{c: stats(part[c]) for c in ["equal_weight", "long_only_net", "long_short_net"]}}
    DOCS.mkdir(exist_ok=True)
    pd.DataFrame(wts).T.to_csv(DOCS / "backtest_v2_weights.csv")
    (DOCS / "backtest_v2_results.json").write_text(json.dumps(out, indent=2))
    eq = (1 + res[["equal_weight", "long_only_net", "long_short_net"]]).cumprod(); eq.to_csv(DOCS / "backtest_v2_equity.csv")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ax = eq.plot(figsize=(8, 4), title="Growth of 1 (net of 10 bps costs); dashed line = train/test split")
        ax.axvline(res.index[mid], ls="--", c="gray"); plt.tight_layout(); plt.savefig(DOCS / "backtest_v2_equity.png", dpi=150)
    except ImportError:
        pass
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
