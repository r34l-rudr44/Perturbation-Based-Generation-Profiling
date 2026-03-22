#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HOME/logprob-data-70b"

MODEL="huihui-ai/Llama-3.3-70B-Instruct-abliterated"
QUANT="4bit"
TASKS=60

if [[ "${1:-}" == "fallback" ]]; then
	MODEL="huihui-ai/Qwen2.5-32B-Instruct-abliterated"
	QUANT="none"
	echo "|  Fallback: Qwen2.5-32B Abliterated (bf16)   |"
	echo "|  ~65GB VRAM, no quantization needed         |"
else
	echo "|  70B Abliterated — 4-bit NF4                 |"
	echo "|  GPU capped at 80GiB + CPU offload           |"
	echo "|  If OOM -> bash run_70b.sh fallback          |"
fi

source ~/logprob-env/bin/activate 2>/dev/null || true

python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
free = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | {free:.1f} GB')
import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')
"

echo ""
echo "--- STEP 1/3: Collecting ($TASKS tasks x 4 runs) ---"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	python3 "$SCRIPT_DIR/collect_logprobs.py" \
	--model "$MODEL" \
	--num-tasks "$TASKS" \
	--output-dir "$DATA_DIR" \
	--quantize "$QUANT"

echo ""
echo "--- STEP 2/3: Features ---"
python3 "$SCRIPT_DIR/extract_features.py" \
	--input "$DATA_DIR/raw_logprob_data.json" \
	--output "$DATA_DIR/features.csv"

echo ""
echo "--- STEP 3/3: Analysis ---"
python3 "$SCRIPT_DIR/analyze.py" \
	--features "$DATA_DIR/features.csv" \
	--raw "$DATA_DIR/raw_logprob_data.json" \
	--output-dir "$DATA_DIR/plots"

echo ""
echo "Done! View: cd $DATA_DIR/plots && python3 -m http.server 8888"
