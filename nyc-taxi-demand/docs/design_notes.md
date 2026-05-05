# Design notes

Short notes-to-self on the choices behind the streaming and parallel modules added in this version. None of these are deeply original — most are direct applications of stuff covered in W7D2 (data streams) and W9 (map-reduce) — but it's worth writing them down so I don't have to rediscover them next semester.

## Why a min-heap top-K, not sort-then-slice

The obvious way to get the top K zones is to count everything, sort by count descending, and take the first K. That's `O(N log N)` for the sort, where N is the number of distinct zones. With ~260 zones that's nothing, but the same idiom shows up at trip-row level too (think top-K customer IDs across hundreds of millions of rides), and there it actually matters.

A min-heap of size K does it in `O(N log K)`:

- Push the first K items. Done.
- For each subsequent item, compare against the heap's min. If it's bigger, replace the min and re-heapify (`heapq.heapreplace`). Otherwise drop it.
- At the end, the heap holds the K largest items — already organized so the smallest of them is at the root, which makes "is this new thing better than the worst one I'm keeping?" an O(1) check followed by an O(log K) replace.

For K=20 vs N in the millions, log K ≈ 4 vs log N ≈ 23. Roughly a 5–6× constant-factor win. More importantly, it's bounded memory: I never have to materialize a sorted array of all distinct keys.

Tie-breaking: I picked "lex-smaller key wins on score tie." Implemented with a small `_HeapItem` wrapper whose `__lt__` flips the secondary comparison so the lex-larger key floats to the top of the min-heap and gets popped first. Could also have done it with `(score, NegStr(key))` but the wrapper class is easier to read.

## Why multiprocessing instead of threading

Per-file aggregation is CPU-bound: parquet decode, pandas groupby, datetime floor. The GIL makes Python threads useless for that kind of work — they'd serialize on the interpreter even if I had 8 cores idle. `multiprocessing.Pool` sidesteps the GIL by giving each worker its own interpreter.

Cost: the worker function has to be picklable (so it lives at module scope, not inside another function), and there's per-task IPC overhead from pickling the arguments and return value. For one parquet that's a few hundred MB, the IPC cost is dwarfed by the per-file decode + groupby time, so the trade-off pays.

Two small details worth flagging:

- **Cap workers at `min(len(paths), cpu_count())`.** Spawning more workers than files is wasted; spawning more than cores starts thrashing. This matches the W9 "don't oversubscribe" advice.
- **`workers=1` skips the Pool entirely.** Easier to test (no pickling, no interpreter-per-worker), easier to debug with a regular stack trace. The result has to be identical either way, so we use the no-pool path as the reference behavior.

## Why no Selenium / web scraping

I considered automating data download with Selenium or a headless-browser approach, but the NYC TLC publishes the parquets at stable, well-known archive URLs of the form `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`. That means a single `curl` (or `urllib.request.urlretrieve`, which is what `scripts/download_data.py` does) is enough — no DOM, no JS execution, no rate-limit gymnastics, no broken selectors when the site refreshes. Selenium would have added a couple hundred MB of dependencies and a flaky failure mode for zero benefit.

If TLC ever moves the files behind a JS-rendered listing page, this calculation flips, and we'd reach for `requests` first and Selenium only if a real browser is required.
