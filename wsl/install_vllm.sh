#!/usr/bin/env bash
set -euo pipefail

# Reproducible WSL setup for the production BFSI router.
# Creates an isolated vLLM environment under $HOME/bfsi-vllm/.venv.

echo "== NVIDIA / WSL check =="
nvidia-smi

echo "== Install uv if missing =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

VENV_ROOT="${VLLM_ROOT:-$HOME/bfsi-vllm}"
VENV="$VENV_ROOT/.venv"

mkdir -p "$VENV_ROOT"
cd "$VENV_ROOT"

if [ ! -d "$VENV" ]; then
  echo "== Create Python 3.12 environment =="
  uv venv --python 3.12 "$VENV"
fi

source "$VENV/bin/activate"

echo "== Install vLLM =="
uv pip install --upgrade \
  vllm \
  --torch-backend=auto

echo "== Install BitsAndBytes runtime =="
uv pip install --upgrade bitsandbytes

echo "== Install client utilities =="
uv pip install --upgrade httpx openai pydantic

echo
echo "INSTALL COMPLETE"
echo "Environment: $VENV"
echo "Next: ./wsl/start_vllm.sh"
