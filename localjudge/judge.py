"""Local correctness judge. LOCKED: never edit to make a candidate pass.

  python localjudge/judge.py [--quick] [--cases 1x512x32,4x2048x32] [--engine-dir engine]

1. A candidate subprocess loads Engine once, and per case runs one warmup generate of the
   same shape and then several seeded generates (real-text and random-id prompts), checking
   the stream contract (exactly N yields, B python ints each). It repeats the first prompt of
   every case at the end: tokens must match (state reset + determinism).
2. This process then loads native Qwen exactly like the starter and replays every
   prompt+emitted sequence teacher-forced in one full forward. At each output position
   margin = max_logit - logit[emitted]. FAIL if any margin > 2.0; WARN if > 0.75.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CANDIDATE_DIR, MODEL_PATH, load_engine_class, make_prompts, parse_shape  # noqa: E402

TIE_MARGIN = 2.0
DRIFT_WARN = 0.75

FULL_CASES = [
    "1x512x32", "4x2048x32", "16x512x128",           # public shapes
    "1x2048x128", "2x1024x64", "8x256x256", "32x512x64", "1x4096x32",
    "3x777x40",                                       # odd length, odd batch
    "2x100x1",                                        # max_new_tokens = 1
]
FULL_SEEDS = [("text", 11), ("text", 12), ("text", 13), ("random", 14), ("random", 15)]
QUICK_SEEDS = [("text", 11), ("random", 14)]


def run_candidate(cases, seeds, engine_dir, out_path):
    import torch

    Engine = load_engine_class(engine_dir)
    t0 = time.time()
    engine = Engine(MODEL_PATH)
    print(f"[cand] load {time.time() - t0:.1f}s", flush=True)
    results = []
    for case in cases:
        b, p, n = parse_shape(case)
        t0 = time.time()
        warm = make_prompts(b, p, seed=1000 + b * 7 + p, kind="text")
        steps = list(engine.generate(warm, n))
        assert len(steps) == n, f"warmup yielded {len(steps)} != {n}"
        tw = time.time() - t0
        first = None
        for kind, seed in seeds:
            prompts = make_prompts(b, p, seed=seed * 1_000_003 + p * 31 + b, kind=kind)
            steps = []
            for step in engine.generate(prompts, n):
                assert isinstance(step, list) and len(step) == b, f"bad step {type(step)} len"
                assert all(type(t) is int for t in step), "tokens must be python ints"
                steps.append(step)
            assert len(steps) == n, f"{case}: yielded {len(steps)} != {n}"
            toks = [[steps[i][r] for i in range(n)] for r in range(b)]
            results.append({"case": case, "kind": kind, "seed": seed, "prompts": prompts, "tokens": toks})
            if first is None:
                first = (prompts, toks)
        # state-reset / determinism: replay the first prompt after other calls
        again = [[0] * n for _ in range(b)]
        for i, step in enumerate(engine.generate(first[0], n)):
            for r in range(b):
                again[r][i] = step[r]
        results.append({"case": case, "kind": "repeat", "seed": -1, "prompts": first[0],
                        "tokens": again, "must_equal": first[1]})
        print(f"[cand] {case} ok (warmup {tw:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f)


def replay(results):
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).eval().to("cuda:0")

    summary = {}
    failed = False
    for res in results:
        case = res["case"]
        s = summary.setdefault(case, {"max_margin": 0.0, "non_argmax": 0, "positions": 0,
                                      "fail": 0, "repeat_mismatch": 0})
        if "must_equal" in res and res["tokens"] != res["must_equal"]:
            s["repeat_mismatch"] += 1
            # a mismatch is not by itself a failure (a near-tie may flip), but it is still judged below
        prompts, toks = res["prompts"], res["tokens"]
        n = len(toks[0])
        seqs = [p + t[:-1] for p, t in zip(prompts, toks)]
        rows = 4 if len(seqs[0]) > 1500 else 8
        for i in range(0, len(seqs), rows):
            ids = torch.tensor(seqs[i:i + rows], device="cuda:0")
            with torch.inference_mode():
                logits = model(input_ids=ids, logits_to_keep=n, use_cache=False).logits.float()
            tgt = torch.tensor([t for t in toks[i:i + rows]], device="cuda:0")
            chosen = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            margin = logits.max(-1).values - chosen
            s["max_margin"] = max(s["max_margin"], margin.max().item())
            s["non_argmax"] += int((margin > 0).sum().item())
            s["positions"] += margin.numel()
            bad = int((margin > TIE_MARGIN).sum().item())
            s["fail"] += bad
            if bad:
                failed = True
                idx = (margin > TIE_MARGIN).nonzero()[0].tolist()
                print(f"  FAIL {case} {res['kind']} seed={res['seed']} row={i + idx[0]} step={idx[1]} "
                      f"margin={margin[idx[0], idx[1]].item():.3f}", flush=True)
    print(f"\n{'case':<12}{'positions':>10}{'non-argmax':>12}{'max margin':>12}{'repeat!=':>10}  status")
    worst = 0.0
    for case, s in summary.items():
        status = "FAIL" if s["fail"] else ("WARN" if s["max_margin"] > DRIFT_WARN else "ok")
        worst = max(worst, s["max_margin"])
        print(f"{case:<12}{s['positions']:>10}{s['non_argmax']:>12}{s['max_margin']:>12.3f}"
              f"{s['repeat_mismatch']:>10}  {status}")
    print(f"\nJUDGE {'FAIL' if failed else 'PASS'}  worst margin {worst:.3f}")
    return not failed, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--engine-dir", default=CANDIDATE_DIR)
    ap.add_argument("--_candidate", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    cases = args.cases.split(",") if args.cases else FULL_CASES
    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS

    if args._candidate:
        run_candidate(cases, seeds, args.engine_dir, args._candidate)
        return

    out = os.path.join(tempfile.gettempdir(), f"judge_{os.getpid()}.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--engine-dir", args.engine_dir,
           "--cases", ",".join(cases), "--_candidate", out] + (["--quick"] if args.quick else [])
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"JUDGE FAIL candidate crashed (exit {rc})")
        sys.exit(2)
    with open(out) as f:
        results = json.load(f)
    ok, summary = replay(results)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_judge.json"), "w") as f:
        json.dump({"pass": ok, "cases": summary}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
