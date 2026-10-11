import json
import pandas as pd, streamlit as st
import engine

ROOT, DOCS = engine.ROOT, engine.ROOT / "docs"
st.set_page_config(page_title="AI/NLP Risk Engine", layout="wide")
st.title("AI/NLP Risk Engine + Module A (sentiment-driven rebalancing)")
st.caption("Reads cached results, so it opens instantly. Models load only if you use the 'Analyze text' tab.")


@st.cache_data
def signals():
    p = ROOT / "data" / "signals_cache.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else pd.DataFrame()


def jread(name):
    p = DOCS / name
    return json.loads(p.read_text()) if p.exists() else None


def cread(*names):
    for n in names:
        if (DOCS / n).exists():
            return pd.read_csv(DOCS / n, index_col=0, parse_dates=True)


t1, t2, t3, t4 = st.tabs(["Risk signals", "Analyze text", "Index weights", "Backtest evidence"])

with t1:
    df = signals()
    if df.empty:
        st.warning("data/signals_cache.json not found. Run: python src/ingest.py build")
    else:
        c = st.columns(3)
        srcs = sorted(df.source.unique())
        src = c[0].multiselect("Source", srcs, default=srcs)
        cos = c[1].multiselect("Company", sorted(df.company.dropna().unique()))
        mn = c[2].slider("Minimum impact", 1.0, 10.0, 1.0, 0.5)
        f = df[df.source.isin(src) & (df.impact_score >= mn)]
        if cos:
            f = f[f.company.isin(cos)]
        m = st.columns(4)
        m[0].metric("Signals", len(f))
        m[1].metric("Avg sentiment", round(f.sentiment_score.mean(), 2) if len(f) else 0)
        m[2].metric("Avg impact", round(f.impact_score.mean(), 1) if len(f) else 0)
        m[3].metric("High impact (>7)", int((f.impact_score > 7).sum()))
        a, b = st.columns(2)
        a.caption("Signals by event class"); a.bar_chart(f.event_class.value_counts())
        b.caption("Average sentiment by company"); b.bar_chart(f.groupby("company").sentiment_score.mean())
        cols = ["source", "company", "date", "sentiment_score", "event_class", "event_confidence", "impact_score", "text"]
        st.dataframe(f.sort_values("impact_score", ascending=False)[cols])

with t2:
    txt = st.text_area("Paste a headline or tweet", "Fed signals rate hike as inflation surges; banks brace for slowdown")
    kind = st.radio("Source type", ["news", "twitter"], horizontal=True)
    if st.button("Analyze") and txt.strip():
        with st.spinner("Loading models on first use (can take a minute)..."):
            r = engine.analyze({"id": 0, "t": 0, "source": kind, "company": None, "text": txt})
            backend = engine.sentiment_backend()
        c = st.columns(4)
        c[0].metric("Sentiment (-1..1)", r["sentiment_score"]); c[1].metric("Event class", r["event_class"])
        c[2].metric("Event confidence", r["event_confidence"]); c[3].metric("Impact (1-10)", r["impact_score"])
        st.write("Tickers found:", ", ".join(r["tickers"]) or "none")
        if r.get("policy_adjusted"):
            st.info(f"Rule-based policy adjustment applied: model score {r['sentiment_raw']} was nudged to {r['sentiment_score']} "
                    "(tax/tariff/ban cue with a weak model score). This rule is a heuristic, not validated.")
        if backend != "finbert":
            st.info("Transformer models are not installed, so the keyword fallback was used. "
                    "Install requirements-rebuild.txt for FinBERT and zero-shot classification.")

with t3:
    w = cread("backtest_v2_weights.csv", "backtest_weights.csv")
    if w is None:
        st.warning("No weights found. Run: python src/backtest_v2.py")
    else:
        st.caption("Long-only tilt: index weights over time (rebalanced every Friday)")
        st.area_chart(w)
        st.caption(f"Latest weights (equal weight would be {1 / len(w.columns):.1%} each)")
        st.bar_chart(w.iloc[-1].sort_values(ascending=False).rename("weight"))

with t4:
    r = jread("backtest_v2_results.json")
    if r is None:
        st.warning("No results found. Run: python src/backtest_v2.py")
    else:
        st.write(f"{r['stocks']} stocks, {r['tweets_used']:,} tweets (about {r['tweets_per_stock_week']} per stock-week), "
                 f"costs {r['cost_bps']:.0f} bps, settings fixed in advance (no tuning).")
        per = ("all", "train_first_half", "test_second_half")
        st.dataframe(pd.DataFrame([{"period": p, "strategy": v, **{k: r[p][v][k] for k in ("total_return", "sharpe", "max_drawdown")}}
                                   for p in per for v in ("equal_weight", "long_only_net", "long_short_net")]))
        st.dataframe(pd.DataFrame({p: {"mean rank IC": r[p]["mean_rank_ic"], "t-stat": r[p]["ic_t_stat"]} for p in per}))
        t = r["test_second_half"]["ic_t_stat"]
        if t is not None and abs(t) > 2:
            st.success("Sentiment IC in the test half is statistically significant (|t| > 2).")
        else:
            st.info("Sentiment IC in the test half is not statistically significant (|t| < 2): no reliable evidence "
                    "that sentiment predicts next-week returns in this sample.")
        eq = cread("backtest_v2_equity.csv")
        if eq is not None:
            st.line_chart(eq)
