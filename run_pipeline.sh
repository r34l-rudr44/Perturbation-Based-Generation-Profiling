#!/bin/bash
set -euo pipefail

MODEL="${1:---model}"
if [[ "$MODEL" == "--model" ]]; then
	MODEL="huihui-ai/Huihui-Qwen3.5-9B-abliterated"
	NUM_TASKS=5
	while [[ $# -gt 0 ]]; do
		case $1 in
		--model)
			MODEL="$2"
			shift 2
			;;
		--num-tasks)
			NUM_TASKS="$2"
			shift 2
			;;
		*) shift ;;
		esac
	done
else
	MODEL="huihui-ai/Huihui-Qwen3.5-9B-abliterated"
	NUM_TASKS=5
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HOME/logprob-data"
PLOTS_DIR="$DATA_DIR/plots"

echo "|  Logprob Control Monitor — Full Pipeline"
echo "|  Model:      $MODEL"
echo "|  Tasks:      $NUM_TASKS"
echo "|  Data dir:   $DATA_DIR"
echo "|  Plots dir:  $PLOTS_DIR"
echo ""

echo "- STEP 1/3: Collecting logprob data"
START=$(date +%s)

python3 "$SCRIPT_DIR/collect_logprobs.py" \
	--model "$MODEL" \
	--num-tasks "$NUM_TASKS" \
	--output-dir "$DATA_DIR" \
	--top-logprobs 50 \
	--gpu-mem 0.85

ELAPSED=$(($(date +%s) - START))
echo "  Collection took ${ELAPSED}s"

echo ""
echo "- STEP 2/3: Extracting features"

python3 "$SCRIPT_DIR/extract_features.py" \
	--input "$DATA_DIR/raw_logprob_data.json" \
	--output "$DATA_DIR/features.csv"

echo ""
echo "- STEP 3/3: Analysis and plotting"

python3 "$SCRIPT_DIR/analyze.py" \
	--features "$DATA_DIR/features.csv" \
	--raw "$DATA_DIR/raw_logprob_data.json" \
	--output-dir "$PLOTS_DIR"

echo ""
echo "|-  Pipeline complete!"
echo "|  Raw data:  $DATA_DIR/raw_logprob_data.json"
echo "|  Features:  $DATA_DIR/features.csv"
echo "|  Plots:     $PLOTS_DIR/*.png"
echo "|  Effect sizes: $DATA_DIR/plots/effect_sizes.csv"
echo ""
