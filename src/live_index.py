"""Streaming Module A. Same long-only tilt as backtest_v2 (w ~ exp(K*z), z-scored across stocks, same weight
caps), but 'mood' decays continuously with a half-life in hours instead of once per weekly step.

Mood per stock = sum of sentiment_score * impact_score / 10 over recent signals (same as backtest.load_signals),
each signal fading as 0.5 ** (age / half_life). half_life and SD_FLOOR are demo settings, NOT validated.
"""
import numpy as np, pandas as pd
from backtest import cap
from backtest_v2 import K

LIVE_UNIVERSE = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "XOM", "BA", "DIS", "KO", "NFLX", "INTC", "PYPL"]
TICKER_MAP = {"GOOGL": "GOOG"}  # engine's alias table says GOOGL, the tweet dataset says GOOG
SD_FLOOR = 0.5  # z-score denominator floor, so a single headline cannot swing the whole index


class LiveIndex:
    def __init__(self, universe=LIVE_UNIVERSE, half_life_h=6.0, mood=None, last=None):
        self.hl, self.last = half_life_h * 3600, last
        self.mood = pd.Series(0.0, index=list(universe))
        for t, v in (mood or {}).items():
            if t in self.mood.index:
                self.mood[t] = v

    def decay(self, now):
        if self.last is not None:
            self.mood *= 0.5 ** (max(0.0, now - self.last) / self.hl)
        self.last = now

    def add(self, sig, now):
        """Apply one engine signal; returns the index members it moved (empty if none are in the universe)."""
        c = sig["sentiment_score"] * sig["impact_score"] / 10 * 0.5 ** (max(0.0, now - sig.get("ts", now)) / self.hl)
        names = [sig.get("company")] + list(sig.get("tickers") or [])
        hit = [t for t in dict.fromkeys(TICKER_MAP.get(x, x) for x in names if x) if t in self.mood.index]
        for t in hit:
            self.mood[t] += c
        return hit

    def weights(self):
        z = (self.mood - self.mood.mean()) / max(self.mood.std(), SD_FLOOR)
        return cap(np.exp(K * z) / np.exp(K * z).sum())
