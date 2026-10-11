"""Real-time pipeline: poll sources -> NLP engine -> structured signals -> Module A weights -> dashboard.

  python src/stream.py                 LIVE: Google News + Reddit, polled every 120 s
  python src/stream.py --replay        OFFLINE DEMO: cached tweets/news, new items every 3 s (tagged 'replay')
  python src/stream.py --fast          keyword NLP backend (no model download); for quick tests
  python src/stream.py --reset         delete previous live outputs first
  python src/stream.py --cycles 1      one poll, then exit (smoke test)

Outputs, read by the dashboard's 'Live feed' tab and usable by any downstream app:
  outputs/live_signals.jsonl   one structured signal per line (append-only)
  outputs/live_weights.csv     index weights after every poll
  outputs/live_state.json      heartbeat, source health, mood
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

if "--fast" in sys.argv:  # must be set before engine is imported
    os.environ["NLP_BACKEND"] = "lexicon"

import engine, live_feed
from live_index import LiveIndex, LIVE_UNIVERSE


def _write_state(state):
    tmp = live_feed.STATE.with_suffix(".tmp")  # atomic swap: the dashboard never reads a half-written file
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, live_feed.STATE)


def _append_weights(w):
    cols = ",".join(w.index)
    if live_feed.WEIGHTS.exists() and live_feed.WEIGHTS.read_text(encoding="utf-8").split("\n", 1)[0] != "ts," + cols:
        live_feed.WEIGHTS.unlink()  # universe changed since the last run: start a fresh history
    new = not live_feed.WEIGHTS.exists()
    with open(live_feed.WEIGHTS, "a", encoding="utf-8") as f:
        if new:
            f.write("ts," + cols + "\n")
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + "," + ",".join(f"{x:.5f}" for x in w) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--sources", default="gnews,social", help="comma list from: gnews, yahoo, social")
    ap.add_argument("--universe", default=",".join(LIVE_UNIVERSE))
    ap.add_argument("--interval", type=float, help="seconds between polls (default 120 live, 3 replay)")
    ap.add_argument("--batch", type=int, default=3, help="replay: items per tick")
    ap.add_argument("--max-new", type=int, default=40, help="max items analysed per poll (zero-shot is slow on CPU)")
    ap.add_argument("--max-age-h", type=float, default=12, help="ignore items older than this")
    ap.add_argument("--half-life-h", type=float, default=6)
    ap.add_argument("--cycles", type=int, default=0, help="0 = run until Ctrl+C")
    a = ap.parse_args(argv)
    a.interval = a.interval or (3 if a.replay else 120)
    universe, sources = a.universe.split(","), a.sources.split(",")
    try:
        sys.stdout.reconfigure(errors="replace")  # tweets/Reddit contain emoji; Windows consoles are cp1252
    except Exception:
        pass

    live_feed.OUT.mkdir(exist_ok=True)
    if a.reset:
        for p in (live_feed.SIGNALS, live_feed.WEIGHTS, live_feed.STATE):
            p.unlink(missing_ok=True)
    old = {}
    try:
        old = json.loads(live_feed.STATE.read_text(encoding="utf-8"))
    except Exception:
        pass
    resume = old.get("universe") == universe and old.get("half_life_h") == a.half_life_h
    idx = LiveIndex(universe, a.half_life_h, old.get("mood") if resume else None, old.get("last") if resume else None)
    seen, total = set(), (old.get("total", 0) if resume else 0)
    if live_feed.SIGNALS.exists():
        for line in live_feed.SIGNALS.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                pass
    replay = live_feed.Replay(universe) if a.replay else None

    print("Loading NLP models (first run downloads them; --fast skips this)...")
    engine.analyze({"id": "warmup", "t": 0, "source": "news", "company": None, "text": "warm up"})
    backend = engine.sentiment_backend()
    mode = "replay" if a.replay else "live"
    print(f"backend={backend} mode={mode} universe={len(universe)} stocks, poll every {a.interval:g}s. Ctrl+C to stop.")

    n = 0
    try:
        while True:
            t0 = time.time()
            n += 1
            try:
                if replay:
                    items, status = replay.next(a.batch), {"replay": {"items": a.batch, "errors": []}}
                else:
                    items, status = live_feed.collect(sources, universe)
                now = time.time()
                fresh = {i["id"]: i for i in items if i["id"] not in seen and now - i["ts"] <= a.max_age_h * 3600}
                fresh = sorted(fresh.values(), key=lambda i: i["ts"])[:a.max_new]  # leftovers are picked up next poll
                idx.decay(now)
                out = []
                for it, s in zip(fresh, engine.sentiment_many([i["text"] for i in fresh]) if fresh else []):
                    r = engine.analyze(it, s)
                    r["affected"] = idx.add(r, now)
                    out.append(r)
                    seen.add(it["id"])
                    print(f"{datetime.now():%H:%M:%S} {r['source']:7} {str(r['company']):5} sent={r['sentiment_score']:+.2f} "
                          f"{r['event_class']:<28} impact={r['impact_score']:<4} -> {','.join(r['affected']) or '-':<10} {r['text'][:60]}")
                if out:
                    with open(live_feed.SIGNALS, "a", encoding="utf-8") as f:
                        f.writelines(json.dumps(r) + "\n" for r in out)
                total += len(out)
                w = idx.weights()
                _append_weights(w)
                _write_state({"mode": mode, "backend": backend, "universe": universe, "half_life_h": a.half_life_h,
                              "interval": a.interval, "updated": time.time(), "cycle": n, "total": total,
                              "new_last_poll": len(out), "sources": status, "mood": idx.mood.round(5).to_dict(),
                              "last": idx.last})
                bad = {k: v["errors"] for k, v in status.items() if v["errors"]}
                print(f"[poll {n}] {len(items)} fetched, {len(out)} new" + (f", source errors: {bad}" if bad else ""))
            except KeyboardInterrupt:
                raise
            except Exception as e:  # keep the pipeline alive through a bad poll
                print(f"[poll {n}] failed: {type(e).__name__}: {e}")
            if a.cycles and n >= a.cycles:
                break
            time.sleep(max(1.0, a.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
