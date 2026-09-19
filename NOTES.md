# NOTES — Qwen3-4B decode engine

## Platform baseline (unchanged starter)
Run `ad765af3-729d-411f-bfcf-209c96dd2af7` (official, commit 3e18b72), succeeded, **score 127.29 tok/s** (geomean over 6 hidden workloads; every run is official).
Container: NVIDIA H100 80GB HBM3, driver 580.95.05, Python 3.11.5, gVisor kernel.

| shape | tok/s | total ms (p50) | TTFT ms | TPOT ms | native TTFT | native TPOT | peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| public-0 b1×512→32 | 27.12 | 1180.0 | 43.47 | 36.64 | 42.00 | 34.88 | 10.28 GB |
| public-1 b4×2048→32 | 92.38 | 1385.6 | 200.73 | 38.24 | 200.37 | 37.77 | 12.28 GB |
| public-2 b16×512→128 | 410.26 | 4991.9 | 190.22 | 37.81 | 189.90 | 37.98 | 12.28 GB |

Observations: TPOT ≈ 37 ms regardless of batch → decode is purely overhead bound on the platform
(bandwidth floor is ~2.5 ms). Gates: our TTFT ≤ 1.10× native TTFT (~46 ms at b1×512, ~220 ms at b4×2048).

Note: the dryft CLI needs `DRYFT_API=https://htn.dryft.ai` (default endpoint returns HTTP 403).
