#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/content/hf-cache
export TOKENIZERS_PARALLELISM=false
python -m pip install -q --no-deps bitsandbytes
python /content/colab_qwen_refusal_lora.py > /content/qwen-lora.log 2>&1
cd /content
tar -czf qwen-authorized-lora.tar.gz qwen-authorized-lora
