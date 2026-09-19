"""Shared helpers for the local judge and bench. LOCKED: the optimizer never edits this.

Mirrors the platform: prompts are token ids (real text via the checkpoint's tokenizer, or
random ids), every sequence in a batch has the same length, fresh prompts every call.
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys

MODEL_PATH = os.environ.get("QWEN_PATH", os.path.expanduser("~/qwen3-4b"))
CORPUS = os.environ.get("CORPUS", os.path.expanduser("~/corpus/text.txt"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATE_DIR = os.environ.get("ENGINE_DIR", os.path.join(ROOT, "engine"))
NATIVE_DIR = os.path.join(ROOT, "localjudge", "native_engine")

_corpus_ids = None


def load_engine_class(engine_dir: str):
    """Import engine.py from engine_dir the way the platform does: dir on sys.path."""
    engine_dir = os.path.abspath(engine_dir)
    sys.path.insert(0, engine_dir)
    spec = importlib.util.spec_from_file_location("engine", os.path.join(engine_dir, "engine.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = mod
    spec.loader.exec_module(mod)
    return mod.Engine


def _corpus():
    global _corpus_ids
    if _corpus_ids is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        with open(CORPUS, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        _corpus_ids = tok(text, add_special_tokens=False)["input_ids"]
    return _corpus_ids


def make_prompts(batch: int, length: int, seed: int, kind: str = "text") -> list[list[int]]:
    rng = random.Random(seed)
    if kind == "random":
        return [[rng.randrange(0, 151643) for _ in range(length)] for _ in range(batch)]
    ids = _corpus()
    out = []
    for _ in range(batch):
        start = rng.randrange(0, len(ids) - length - 1)
        out.append(list(ids[start:start + length]))
    return out


def parse_shape(s: str) -> tuple[int, int, int]:
    """'4x2048x32' -> (batch, prompt, out)."""
    b, p, n = (int(x) for x in s.lower().split("x"))
    return b, p, n
