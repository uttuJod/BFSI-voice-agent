#!/usr/bin/env bash
set -euo pipefail

VENV="${VLLM_VENV:-$HOME/bfsi-vllm/.venv}"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "Missing vLLM environment: $VENV"
  echo "Run ./wsl/install_vllm.sh first."
  exit 1
fi

source "$VENV/bin/activate"

# Keep the benchmarked native sampler path.
export VLLM_USE_FLASHINFER_SAMPLER=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
LORA_PATH="${LORA_PATH:-$PROJECT_ROOT/models/qwen35_customer_support_FINAL_FROZEN}"

if [ ! -f "$LORA_PATH/adapter_config.json" ]; then
  echo "LoRA adapter metadata not found:"
  echo "  $LORA_PATH/adapter_config.json"
  echo
  echo "The public repository intentionally excludes adapter_model.safetensors."
  echo "Set LORA_PATH to a local complete adapter directory before starting vLLM."
  exit 1
fi

if [ ! -f "$LORA_PATH/adapter_model.safetensors" ]; then
  echo "LoRA weight file not found:"
  echo "  $LORA_PATH/adapter_model.safetensors"
  echo
  echo "The public repository intentionally excludes model weights."
  echo "Set LORA_PATH to your local complete adapter directory."
  exit 1
fi

echo "== vLLM preflight =="

python - <<'PY'
import os
import vllm

print("vLLM version:", vllm.__version__)
print(
    "VLLM_USE_FLASHINFER_SAMPLER:",
    os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
)

if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
    raise RuntimeError("Native sampler configuration is not active.")

from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
print("PASS: built-in BitsAndBytes path imports.")
PY

echo
echo "Project root : $PROJECT_ROOT"
echo "LoRA adapter : $LORA_PATH"
echo "Endpoint     : http://127.0.0.1:8001/v1"
echo "LoRA name    : bfsi-router"
echo

exec vllm serve unsloth/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name qwen35-base \
  --language-model-only \
  --quantization bitsandbytes \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 32 \
  --lora-modules "bfsi-router=$LORA_PATH" \
  --chat-template "$LORA_PATH/chat_template.jinja" \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching
