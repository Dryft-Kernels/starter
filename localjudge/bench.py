"""Local benchmark mirroring the platform protocol. LOCKED.

  python localjudge/bench.py [--shapes 1x512x32,...] [--native] [--refresh-native] [--samples 5]

Per shape, a FRESH process: construct Engine, one warmup generate of the same shape, then
`samples` timed generates with fresh prompts. TTFT = call -> first yield, TPOT = rest / (N-1),
total = call -> stream exhausted, tok/s = B*N/total. Medians are reported, spread = (max-min)/median.
Native (starter) numbers are cached in localjudge/native_results.json.
Prints Hotpath JSON (metric tokens_per_s; aggregate samples = per-sample geomean across shapes).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CANDIDATE_DIR, MODEL_PATH, NATIVE_DIR, load_engine_class, make_prompts, parse_shape  # noqa: E402

DEFAULT_SHAPES = ["1x512x32", "4x2048x32", "16x512x128", "1x2048x128", "8x256x256", "32x512x64"]
NATIVE_CACHE = os.path.join(HERE, "native_results.json")
FAIL_RATIO = 0.95  # locally treat latency ratio above this as a failure (platform gate is 1.10)


def worker(engine_dir, shape, samples):
    import torch

    b, p, n = parse_shape(shape)
    Engine = load_engine_class(engine_dir)
    t0 = time.perf_counter()
    engine = Engine(MODEL_PATH)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in engine.generate(make_prompts(b, p, seed=999, kind="text"), n):
        pass
    warm_s = time.perf_counter() - t0
    rows = []
    for i in range(samples):
        prompts = make_prompts(b, p, seed=5000 + i, kind="text")
        gc.collect()
        torch.cuda.synchronize()
        count = 0
        t0 = time.perf_counter()
        ttft = None
        for step in engine.generate(prompts, n):
            count += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
        total = time.perf_counter() - t0
        assert count == n
        tpot = (total - ttft) / (n - 1) if n > 1 else 0.0
        rows.append({"ttft": ttft, "tpot": tpot, "total": total, "tps": b * n / total})
    free, dev_total = torch.cuda.mem_get_info()
    out = {"shape": shape, "load_s": load_s, "warm_s": warm_s, "samples": rows,
           "peak_reserved": torch.cuda.max_memory_reserved(), "device_used": dev_total - free,
           "device_total": dev_total}
    print("BENCH_RESULT " + json.dumps(out), flush=True)


def run_shape(engine_dir, shape, samples):
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", shape, "--engine-dir", engine_dir,
           "--samples", str(samples)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT "):
            return json.loads(line[len("BENCH_RESULT "):])
    sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-5000:])
    raise RuntimeError(f"{shape}: worker failed (exit {proc.returncode})")


def med(r, k):
    return statistics.median(s[k] for s in r["samples"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--engine-dir", default=CANDIDATE_DIR)
    ap.add_argument("--refresh-native", action="store_true")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._worker:
        worker(args.engine_dir, args._worker, args.samples)
        return

    shapes = args.shapes.split(",")
    native = json.load(open(NATIVE_CACHE)) if os.path.exists(NATIVE_CACHE) else {}
    for s in shapes:
        if s not in native or args.refresh_native:
            print(f"[native] {s} ...", flush=True)
            native[s] = run_shape(NATIVE_DIR, s, 5)
            json.dump(native, open(NATIVE_CACHE, "w"), indent=1)

    results, ok, speedups = {}, True, []
    hdr = (f"{'shape':<12}{'tok/s':>9}{'native':>9}{'speedup':>9}{'TTFT ms':>9}{'TTFTr':>7}"
           f"{'TPOT ms':>9}{'TPOTr':>7}{'spread':>8}{'peakGiB':>8}{'load+warm':>10}")
    for s in shapes:
        print(f"[cand] {s} ...", flush=True)
        r = results[s] = run_shape(args.engine_dir, s, args.samples)
    print("\n" + hdr)
    for s in shapes:
        r, nr = results[s], native[s]
        b, p, n = parse_shape(s)
        tps, ntps = med(r, "tps"), med(nr, "tps")
        ttft_r = med(r, "ttft") / med(nr, "ttft")
        tpot_r = med(r, "tpot") / med(nr, "tpot") if n > 1 else 0.0
        totals = [x["total"] for x in r["samples"]]
        spread = (max(totals) - min(totals)) / statistics.median(totals)
        peak = r["device_used"] / 2**30
        flags = []
        if ttft_r > FAIL_RATIO: flags.append("TTFT")
        if tpot_r > FAIL_RATIO: flags.append("TPOT")
        if spread > 0.10: flags.append("SPREAD")
        if r["device_used"] > 0.80 * r["device_total"]: flags.append("MEM")
        if r["load_s"] + r["warm_s"] > 200: flags.append("LOAD")
        ok &= not flags
        speedups.append(tps / ntps)
        print(f"{s:<12}{tps:>9.1f}{ntps:>9.1f}{tps / ntps:>8.2f}x{med(r, 'ttft') * 1e3:>9.1f}{ttft_r:>7.2f}"
              f"{med(r, 'tpot') * 1e3:>9.2f}{tpot_r:>7.2f}{spread * 100:>7.1f}%{peak:>8.1f}"
              f"{r['load_s'] + r['warm_s']:>9.0f}s  {' '.join(flags)}")
    geo = math.exp(sum(math.log(x) for x in speedups) / len(speedups))
    print(f"\nGEOMEAN speedup {geo:.2f}x   {'BENCH OK' if ok else 'BENCH FLAGGED'}")
    json.dump({"results": results, "geomean_speedup": geo, "ok": ok},
              open(os.path.join(HERE, "last_bench.json"), "w"), indent=1)

    k = min(len(results[s]["samples"]) for s in shapes)
    agg = [math.exp(sum(math.log(results[s]["samples"][i]["tps"]) for s in shapes) / len(shapes)) for i in range(k)]
    workloads = {s: [x["tps"] for x in results[s]["samples"]] for s in shapes}
    try:
        from hotpath.benchlib import emit
        emit(agg, metric="tokens_per_s", higher_is_better=True, workloads=workloads)
    except ImportError:
        print(json.dumps({"hotpath_benchmark": 1, "metric": "tokens_per_s", "higher_is_better": True,
                          "samples": agg, "workloads": workloads}))


if __name__ == "__main__":
    main()
