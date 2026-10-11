"""Live data sources for the NLP risk engine. All free, no API key.

  gnews   Google News RSS: one query per company + a few macro/sector queries   -> source "news"
  yahoo   Yahoo Finance headline RSS, one request per ticker                    -> source "news"
  social  Reddit search RSS (r/stocks, r/investing, r/wallstreetbets)           -> source "social"
  replay  cached tweets/news from data/ played back at speed (offline fallback) -> feed "replay"

Fetchers never raise: a source that is down is reported in its status dict and skipped for that cycle.
Every item is engine-ready: {id, t, source, company, date, text, url, ts, feed}.
"""
import hashlib, html, json, random, re, time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import pandas as pd, requests
import engine

OUT = engine.ROOT / "outputs"
SIGNALS, WEIGHTS, STATE = OUT / "live_signals.jsonl", OUT / "live_weights.csv", OUT / "live_state.json"

NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOG": "Google", "AMZN": "Amazon", "META": "Meta Platforms",
         "TSLA": "Tesla", "NVDA": "Nvidia", "JPM": "JPMorgan", "XOM": "Exxon Mobil", "BA": "Boeing",
         "DIS": "Disney", "KO": "Coca-Cola", "NFLX": "Netflix", "INTC": "Intel", "PYPL": "PayPal",
         "AMD": "AMD", "COST": "Costco", "CRM": "Salesforce", "VZ": "Verizon"}
MACRO = ["Federal Reserve interest rates", "tariffs", "e-commerce tax", "semiconductor export controls", "oil prices OPEC"]
UA = {"User-Agent": "Mozilla/5.0 (compatible; risk-engine-hackathon/1.0; educational project)"}
GNEWS = "https://news.google.com/rss/search"
YAHOO = "https://feeds.finance.yahoo.com/rss/2.0/headline"
REDDIT = "https://www.reddit.com/r/stocks+investing+wallstreetbets+StockMarket/search.rss"


def _get(url, params=None, timeout=10):
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.content


def _when(s):
    """RSS (RFC 822) or Atom (ISO 8601) timestamp -> aware UTC datetime, or None."""
    for parse in (parsedate_to_datetime, lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))):
        try:
            d = parse(s)
            return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        except Exception:
            pass
    return None


def _strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def parse_feed(xml):
    """RSS 2.0 <item> or Atom <entry> -> [{title, body, link, when}] (namespace-agnostic)."""
    out = []
    for el in ET.fromstring(xml).iter():
        if el.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        f = {}
        for c in el:
            f.setdefault(c.tag.rsplit("}", 1)[-1], c)

        def txt(*names):
            return next((f[n].text.strip() for n in names if n in f and (f[n].text or "").strip()), "")
        link = (f["link"].get("href") if "link" in f else None) or txt("link")
        out.append({"title": html.unescape(txt("title")), "body": _strip(txt("description", "content", "summary")),
                    "link": link, "when": _when(txt("pubDate", "updated", "published"))})
    return out


def _collect(jobs, source, body=False, pause=0.5):
    """jobs: [(company_or_None, url, params)] -> (items, errors). One bad request never aborts the rest."""
    items, errs = [], []
    for company, url, params in jobs:
        try:
            for e in parse_feed(_get(url, params)):
                title = re.sub(r"\s+-\s+[^-]{2,40}$", "", e["title"]) if source == "news" and url == GNEWS else e["title"]
                if len(title) < 15:
                    continue
                text = f"{title}. {e['body'][:300]}" if body and e["body"] else title
                when = e["when"] or datetime.now(timezone.utc)
                key = hashlib.sha1(re.sub(r"\W+", " ", title.lower()).strip().encode()).hexdigest()[:12]
                items.append({"id": f"{source}-{key}", "t": 99, "source": source, "company": company,
                              "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "ts": when.timestamp(),
                              "text": text, "url": e["link"], "feed": "live"})
        except Exception as ex:
            errs.append(f"{type(ex).__name__}: {str(ex)[:70]}")
        time.sleep(pause)
    return items, list(dict.fromkeys(errs))[:3]


def fetch_gnews(universe):
    gn = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
    jobs = [(t, GNEWS, {"q": f'"{NAMES.get(t, t)}" stock when:1d', **gn}) for t in universe]
    jobs += [(None, GNEWS, {"q": f"{q} when:1d", **gn}) for q in MACRO]
    return _collect(jobs, "news")


def fetch_yahoo(universe):
    return _collect([(t, YAHOO, {"s": t, "region": "US", "lang": "en-US"}) for t in universe], "news", body=True)


def fetch_social(universe):
    jobs = [(t, REDDIT, {"q": f'"{NAMES.get(t, t)}"', "restrict_sr": 1, "sort": "new", "t": "day"}) for t in universe]
    return _collect(jobs, "social", body=True, pause=1.0)  # Reddit rate-limits harder


FETCHERS = {"gnews": fetch_gnews, "yahoo": fetch_yahoo, "social": fetch_social}


def collect(sources, universe):
    items, status = [], {}
    for name in sources:
        got, errs = FETCHERS[name](universe)
        status[name] = {"items": len(got), "errors": errs}
        items += got
    return items, status


class Replay:
    """Offline fallback: replays cached tweets/news from data/ in random order. Items are stamped with the
    current time and tagged feed='replay' so they can never be mistaken for live data."""
    MAP = {"GOOGL": "GOOG"}

    def __init__(self, universe, seed=7):
        d, rows = engine.ROOT / "data", []
        if (d / "tweets_sample.csv").exists():
            rows += [("twitter", r.company, r.text) for r in pd.read_csv(d / "tweets_sample.csv").itertuples()]
        if (d / "news_cache.json").exists():
            rows += [("news", x["company"], x["text"]) for x in json.loads((d / "news_cache.json").read_text(encoding="utf-8"))]
        self.rows = [(s, self.MAP.get(c, c), t) for s, c, t in rows if self.MAP.get(c, c) in universe]
        if not self.rows:
            raise SystemExit("Replay found no cached items for this universe in data/.")
        random.Random(seed).shuffle(self.rows)
        self.i, self.run = 0, int(time.time())

    def next(self, n):
        now, out = datetime.now(timezone.utc), []
        for _ in range(n):
            s, c, t = self.rows[self.i % len(self.rows)]
            self.i += 1
            out.append({"id": f"replay-{self.run}-{self.i}", "t": 99, "source": s, "company": c,
                        "date": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "ts": now.timestamp(),
                        "text": t, "url": "", "feed": "replay"})
        return out


def read_live(n=300):
    """For the dashboard: (state dict | None, recent signals DataFrame, weights DataFrame). Tolerates partial writes."""
    state, rows = None, []
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        pass
    if SIGNALS.exists():
        for line in SIGNALS.read_text(encoding="utf-8").splitlines()[-n:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    try:
        w = pd.read_csv(WEIGHTS, index_col=0, parse_dates=True).tail(2000)
    except Exception:
        w = pd.DataFrame()
    return state, pd.DataFrame(rows), w
