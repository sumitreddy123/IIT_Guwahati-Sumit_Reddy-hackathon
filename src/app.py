import json, streamlit as st, pandas as pd
import engine, module_a

st.set_page_config(page_title="NLP Risk Engine", layout="wide")
st.title("AI/NLP Risk Engine + Tactical Index Rebalancer")

signals = engine.run()
df = pd.DataFrame(signals)

st.subheader("1. Live text analyzer")
txt = st.text_input("Paste a headline or tweet", "Fed signals rate hike as inflation surges")
if txt:
    r = engine.analyze({"id": 0, "t": 0, "source": "news", "company": None, "text": txt})
    c = st.columns(3)
    c[0].metric("Sentiment", r["sentiment_score"]); c[1].metric("Event", r["event_class"]); c[2].metric("Impact (1-10)", r["impact_score"])

st.subheader("2. Structured risk signals")
st.dataframe(df[["t", "source", "company", "sentiment_score", "event_class", "impact_score", "text"]], use_container_width=True)
st.download_button("Download signals.json", json.dumps(signals, indent=2), "signals.json")

st.subheader("3. Module A: sentiment-driven index weights")
hist = module_a.rebalance(signals)
c1, c2 = st.columns(2)
c1.caption("Weights over time"); c1.area_chart(hist)
c2.caption("Latest weights vs equal weight"); c2.bar_chart(hist.iloc[-1])
st.caption(f"Equal weight = {1/len(module_a.UNIVERSE):.0%}; cap {module_a.W_MAX:.0%}, floor {module_a.W_MIN:.0%}.")
st.line_chart(module_a.prices())
