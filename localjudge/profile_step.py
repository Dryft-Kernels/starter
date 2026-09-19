"""Profile one generate call of the candidate. LOCKED.

  python localjudge/profile_step.py [--shape 1x512x32]

Runs warmup, then profiles a generate call with torch.profiler and prints the top CUDA
kernels, kernel counts per decode step, and GPU time for prefill vs decode.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CANDIDATE_DIR, MODEL_PATH, load_engine_class, make_prompts, parse_shape  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1x512x32")
    ap.add_argument("--engine-dir", default=CANDIDATE_DIR)
    ap.add_argument("--rows", type=int, default=25)
    args = ap.parse_args()
    import torch
    from torch.profiler import ProfilerActivity, profile

    b, p, n = parse_shape(args.shape)
    engine = load_engine_class(args.engine_dir)(MODEL_PATH)
    for _ in engine.generate(make_prompts(b, p, 1), n):
        pass
    prompts = make_prompts(b, p, 2)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in engine.generate(prompts, n):
            pass
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    total_us = sum(e.device_time for e in events) if events and hasattr(events[0], "device_time") else \
        sum(e.cuda_time for e in events)
    print(f"shape {args.shape}: {len(events)} GPU kernels total, {total_us / 1e3:.2f} ms GPU busy, "
          f"~{len(events) / max(n, 1):.0f} kernels/step")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows))


if __name__ == "__main__":
    main()
