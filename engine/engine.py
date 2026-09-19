"""Qwen3 4B greedy decode engine.

Loads the checkpoint with Transformers (so tied weights and dtypes are exactly the
reference's), then runs its own forward: packed QKV and gate/up weights, a static
per-layer KV cache, SDPA flash prefill, a Triton split-K GQA decode attention that reads a
device-side position, and CUDA graphs over both prefill and the decode step. Graphs are
captured on the first call of each shape (the platform's warmup), then replayed.

Arithmetic follows transformers 4.51.3 modeling_qwen3 operation by operation; only the
order of reductions differs.
"""

import gc
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels.decode_attn import DecodeAttention  # noqa: E402
from kernels.rmsnorm import rms_norm_rows  # noqa: E402

DEVICE = "cuda:0"
N_HEADS, N_KV, HEAD_DIM = 32, 8, 128
Q_SIZE, KV_SIZE = N_HEADS * HEAD_DIM, N_KV * HEAD_DIM


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _Layer:
    __slots__ = ("ln1", "w_qkv", "q_norm", "k_norm", "w_o", "ln2", "w_gu", "w_down")


class _State:
    """Everything shape-dependent for one (batch, prompt_len, max_new_tokens)."""

    def __init__(self, eng, B, S, N):
        self.B, self.S, self.N = B, S, N
        cap = S + N
        self.capacity = cap
        self.k_cache = [torch.zeros((B, N_KV, cap, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE) for _ in eng.layers]
        self.v_cache = [torch.zeros((B, N_KV, cap, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE) for _ in eng.layers]
        pos_ids = torch.arange(cap, device=DEVICE)[None]
        dummy = torch.empty(1, dtype=torch.bfloat16, device=DEVICE)
        cos, sin = eng.rotary(dummy, pos_ids)                   # reference rotary module, [1, cap, 128] BF16
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()
        self.ids = torch.zeros((B, S), dtype=torch.int64, device=DEVICE)
        self.tok = torch.zeros((B,), dtype=torch.int64, device=DEVICE)
        self.pos = torch.zeros((1,), dtype=torch.int64, device=DEVICE)
        self.attn = DecodeAttention(B, N_KV, cap, DEVICE, eng.sm_count)
        self.attn_out = torch.empty((B, N_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        self.host = torch.empty((max(N, 1), B), dtype=torch.int64, pin_memory=True)
        self.ids_host = torch.empty((B, S), dtype=torch.int64, pin_memory=True)
        self.events = [torch.cuda.Event() for _ in range(max(N, 1))]
        self.prefill_graph = None
        self.decode_graph = None


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
            ).eval().to(DEVICE)
        )
        base = model.model
        cfg = model.config
        self.eps = cfg.rms_norm_eps
        self.embed = base.embed_tokens.weight
        self.lm_head = model.lm_head.weight                    # tied: same storage as embed
        self.final_norm = base.norm.weight
        self.rotary = base.rotary_emb
        self.layers = []
        with torch.no_grad():
            for layer in base.layers:
                at, mlp = layer.self_attn, layer.mlp
                L = _Layer()
                L.ln1 = layer.input_layernorm.weight
                L.ln2 = layer.post_attention_layernorm.weight
                L.q_norm, L.k_norm = at.q_norm.weight, at.k_norm.weight
                L.w_qkv = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous()
                L.w_o = at.o_proj.weight
                L.w_gu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
                L.w_down = mlp.down_proj.weight
                at.q_proj = at.k_proj = at.v_proj = None
                mlp.gate_proj = mlp.up_proj = None
                self.layers.append(L)
                torch.cuda.empty_cache()
        self.inter = cfg.intermediate_size
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        self.sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        self.state = None

    # ------------------------------------------------------------------ forward pieces
    def _norm(self, x2d, w, heads=1):
        return rms_norm_rows(x2d, w, self.eps, heads)

    def _prefill(self, st):
        """Full prompt forward; writes KV cache [0, S) and the first token into st.tok."""
        B, S = st.B, st.S
        M = B * S
        x = F.embedding(st.ids, self.embed).view(M, -1)       # [M, H]
        cos = st.cos[:S][None, None]                            # [1, 1, S, 128]
        sin = st.sin[:S][None, None]
        for i, L in enumerate(self.layers):
            h = self._norm(x, L.ln1).view(M, -1)
            qkv = F.linear(h, L.w_qkv)                          # [M, 6144]
            q = self._norm(qkv, L.q_norm, N_HEADS).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = self._norm(qkv[:, Q_SIZE:], L.k_norm, N_KV).view(B, S, N_KV, HEAD_DIM).transpose(1, 2)
            v = qkv[:, Q_SIZE + KV_SIZE:].view(B, S, N_KV, HEAD_DIM).transpose(1, 2)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            st.k_cache[i][:, :, :S].copy_(k)
            st.v_cache[i][:, :, :S].copy_(v)
            kr = k[:, :, None].expand(B, N_KV, N_HEADS // N_KV, S, HEAD_DIM).reshape(B, N_HEADS, S, HEAD_DIM)
            vr = v[:, :, None].expand(B, N_KV, N_HEADS // N_KV, S, HEAD_DIM).reshape(B, N_HEADS, S, HEAD_DIM)
            a = F.scaled_dot_product_attention(q.contiguous(), kr, vr, is_causal=True, scale=HEAD_DIM ** -0.5)
            a = a.transpose(1, 2).reshape(M, Q_SIZE)
            x = x + F.linear(a, L.w_o)
            h = self._norm(x, L.ln2).view(M, -1)
            gu = F.linear(h, L.w_gu)
            x = x + F.linear(F.silu(gu[:, :self.inter]) * gu[:, self.inter:], L.w_down)
        last = x.view(B, S, -1)[:, -1]
        h = self._norm(last, self.final_norm).view(B, -1)
        logits = F.linear(h, self.lm_head)
        st.tok.copy_(torch.argmax(logits, dim=-1))

    def _decode_step(self, st):
        """One token per sequence at position st.pos; updates st.tok and advances st.pos."""
        B = st.B
        x = F.embedding(st.tok, self.embed)                     # [B, H]
        cos = st.cos.index_select(0, st.pos)[None]              # [1, 1, 128]
        sin = st.sin.index_select(0, st.pos)[None]
        for i, L in enumerate(self.layers):
            h = self._norm(x, L.ln1).view(B, -1)
            qkv = F.linear(h, L.w_qkv)
            q = self._norm(qkv, L.q_norm, N_HEADS)              # [B, 32, 128]
            k = self._norm(qkv[:, Q_SIZE:], L.k_norm, N_KV)     # [B, 8, 128]
            v = qkv[:, Q_SIZE + KV_SIZE:].view(B, N_KV, HEAD_DIM)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            st.k_cache[i].index_copy_(2, st.pos, k[:, :, None])
            st.v_cache[i].index_copy_(2, st.pos, v[:, :, None])
            a = st.attn(q.contiguous(), st.k_cache[i], st.v_cache[i], st.pos, st.attn_out)
            x = x + F.linear(a.view(B, Q_SIZE), L.w_o)
            h = self._norm(x, L.ln2).view(B, -1)
            gu = F.linear(h, L.w_gu)
            x = x + F.linear(F.silu(gu[:, :self.inter]) * gu[:, self.inter:], L.w_down)
        h = self._norm(x, self.final_norm).view(B, -1)
        logits = F.linear(h, self.lm_head)
        st.tok.copy_(torch.argmax(logits, dim=-1))
        st.pos.add_(1)

    # ------------------------------------------------------------------ graphs
    def _build(self, B, S, N):
        self.state = None
        gc.collect()
        torch.cuda.empty_cache()
        st = _State(self, B, S, N)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._prefill(st)
                if N > 1:
                    st.pos.fill_(S)
                    self._decode_step(st)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._prefill(st)
            st.prefill_graph = g
            if N > 1:
                st.pos.fill_(S)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._decode_step(st)
                st.decode_graph = g
        except Exception as exc:  # fall back to eager rather than fail the run
            print(f"[engine] graph capture failed, running eager: {exc!r}", file=sys.stderr, flush=True)
            st.prefill_graph = st.decode_graph = None
        torch.cuda.synchronize()
        self.state = st
        return st

    # ------------------------------------------------------------------ API
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        N = int(max_new_tokens)
        if N <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        gc_was = gc.isenabled()
        gc.disable()
        try:
            with torch.inference_mode():
                st = self.state
                if st is None or (st.B, st.S, st.N) != (B, S, N):
                    st = self._build(B, S, N)
                st.ids_host.copy_(torch.tensor(input_ids, dtype=torch.int64))
                st.ids.copy_(st.ids_host, non_blocking=True)
                st.pos.fill_(S)
                # prefill -> token 0
                if st.prefill_graph is not None:
                    st.prefill_graph.replay()
                else:
                    self._prefill(st)
                st.host[0].copy_(st.tok, non_blocking=True)
                st.events[0].record()
            for t in range(1, N):
                # enqueue step t before waiting on step t-1, keeping the GPU one step ahead
                with torch.inference_mode():
                    if st.decode_graph is not None:
                        st.decode_graph.replay()
                    else:
                        self._decode_step(st)
                    st.host[t].copy_(st.tok, non_blocking=True)
                    st.events[t].record()
                st.events[t - 1].synchronize()
                yield st.host[t - 1].tolist()
            st.events[N - 1].synchronize()
            yield st.host[N - 1].tolist()
        finally:
            if gc_was:
                gc.enable()
