#!/usr/bin/env bash
# Idempotent setup for a fresh Baseten H100 workstation. Safe to re-run.
#   ssh <host> 'bash -s' < scripts/remote_setup.sh
# Never prints secrets. Everything lands under $HOME.
set -euo pipefail

FORK_URL="${FORK_URL:-https://github.com/Pranoym17/starter.git}"
HOTPATH_URL="${HOTPATH_URL:-https://github.com/Nijjea1/hotpath.git}"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REV="cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_DIR="$HOME/qwen3-4b"
VENV="$HOME/dryft-venv"
HP_VENV="$HOME/hotpath-venv"

log() { echo "[setup] $*"; }

log "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max --format=csv,noheader
log "disk:"; df -h "$HOME" | tail -1

# --- Python 3.11 (uv provides it if the image lacks it) ---------------------------------------
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"

# --- starter repo ------------------------------------------------------------------------------
if [ ! -d "$HOME/dryft/.git" ]; then
  git clone -q "$FORK_URL" "$HOME/dryft"
fi
(cd "$HOME/dryft" && git fetch -q origin)

# --- runtime venv, pinned to the container versions ---------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q --python 3.11 "$VENV"
fi
if ! "$VENV/bin/python" -c 'import torch, triton, transformers, sys; sys.exit(0 if (torch.__version__.startswith("2.5.1") and triton.__version__=="3.1.0" and transformers.__version__=="4.51.3") else 1)' 2>/dev/null; then
  log "installing pinned runtime"
  VIRTUAL_ENV="$VENV" uv pip install -q -r "$HOME/dryft/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
  VIRTUAL_ENV="$VENV" uv pip install -q "huggingface_hub[cli]<0.31" numpy
fi
"$VENV/bin/python" - <<'EOF'
import torch, triton, transformers, sys
print("[setup] python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "triton", triton.__version__, "transformers", transformers.__version__,
      "cuda_available", torch.cuda.is_available())
EOF

# --- checkpoint --------------------------------------------------------------------------------
if [ ! -f "$MODEL_DIR/config.json" ] || ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  log "downloading $MODEL_ID@$MODEL_REV"
  "$VENV/bin/huggingface-cli" download "$MODEL_ID" --revision "$MODEL_REV" --local-dir "$MODEL_DIR" >/dev/null
fi
log "model: $(du -sh "$MODEL_DIR" | cut -f1)"

# --- corpus for real-text prompts (public domain) ----------------------------------------------
mkdir -p "$HOME/corpus"
if [ ! -s "$HOME/corpus/text.txt" ]; then
  for id in 1342 84 2701 11 1661; do  # Austen, Shelley, Melville, Carroll, Doyle
    curl -fsSL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" >> "$HOME/corpus/text.txt" || true
  done
fi
log "corpus bytes: $(wc -c < "$HOME/corpus/text.txt")"

# --- hotpath (separate venv, never in engine/) --------------------------------------------------
if [ ! -d "$HOME/hotpath/.git" ]; then
  git clone -q "$HOTPATH_URL" "$HOME/hotpath" || log "hotpath clone failed (private?)"
fi
if [ -d "$HOME/hotpath" ] && [ ! -x "$HP_VENV/bin/hotpath" ]; then
  uv venv -q --python 3.11 "$HP_VENV"
  VIRTUAL_ENV="$HP_VENV" uv pip install -q -e "$HOME/hotpath" || log "hotpath install failed"
fi

# --- HBM bandwidth -----------------------------------------------------------------------------
"$VENV/bin/python" - <<'EOF'
import torch
n = 2 * 1024**3  # 2 GiB
a = torch.empty(n, dtype=torch.uint8, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize()
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record()
for _ in range(20): b.copy_(a)
e.record(); torch.cuda.synchronize()
ms = s.elapsed_time(e) / 20
print(f"[setup] HBM d2d copy: {2*n/ms/1e9:.0f} GB/s (read+write) over 2 GiB")
EOF
log "done"
