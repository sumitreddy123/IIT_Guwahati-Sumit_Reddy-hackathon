"""Re-run the event classifier on cached signals that are labeled 'Other' (or on everything with --all),
reusing the cached FinBERT scores, so the extended class list does not need a full rebuild.

    python src/reclassify.py          # only items currently 'Other'
    python src/reclassify.py --all    # every item (consistent labels, slower)

Progress is saved every 50 items; items that moved out of 'Other' are skipped if you re-run.
"""
import argparse, collections, json
import engine

CACHE = engine.ROOT / "data" / "signals_cache.json"
KEEP = ("id", "t", "source", "company", "date", "text")


def main(everything):
    items = json.loads(CACHE.read_text())
    before = collections.Counter(i["event_class"] for i in items)
    todo = [k for k, i in enumerate(items) if everything or i["event_class"] == "Other"]
    print(f"{len(todo)} of {len(items)} items to re-run (backend: {engine.sentiment_backend()})")
    for n, k in enumerate(todo, 1):
        i = items[k]
        items[k] = engine.analyze({key: i[key] for key in KEEP if key in i}, i.get("sentiment_raw", i["sentiment_score"]))
        if n % 50 == 0 or n == len(todo):
            CACHE.write_text(json.dumps(items, indent=1))
            print(f"  {n}/{len(todo)} done")
    after = collections.Counter(i["event_class"] for i in items)
    print(f"\n{'class':32}{'before':>8}{'after':>8}")
    for c in sorted(set(before) | set(after), key=lambda c: -after[c]):
        print(f"{c:32}{before[c]:>8}{after[c]:>8}")
    print(f"\n'Other' share: {before['Other'] / len(items):.0%} -> {after['Other'] / len(items):.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true")
    main(ap.parse_args().all)
