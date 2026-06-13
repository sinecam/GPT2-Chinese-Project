#!/bin/bash

source /root/autodl-tmp/envs/gpt2/bin/activate

export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export TORCH_HOME=/root/autodl-tmp/cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/cache
export TMPDIR=/root/autodl-tmp/cache/tmp
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton

echo "GPT-2 environment activated."
echo "Python: $(which python)"
echo "Pip: $(which pip)"
export PYTHONPATH=/root/autodl-tmp/GPT2_Small_Project:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
