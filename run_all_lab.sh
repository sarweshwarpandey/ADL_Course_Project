#!/bin/bash
set -euo pipefail

# ==============================================================================
# End-to-End Lab PC Runner for ADL Course Project (Topic 18: CUIG)
# Executes:
#   1. Official UnlearnCanvas Checkpoint Verification / Download
#   2. Task 18.1: Threshold-Aware Adaptive Model Merging (Style & Object)
#   3. Task 18.2: Adaptive Anchor Assignment for Continual Unlearning (Style & Object)
#   4. High-Resolution Presentation & Report Plot Generation
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "=================================================================="
echo " Step 1: Checking / Downloading UnlearnCanvas Checkpoints"
echo "=================================================================="
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    bash download_checkpoints.sh
else
    echo "Skipping checkpoint download (SKIP_DOWNLOAD=1)."
fi

echo ""
echo "=================================================================="
echo " Step 2: Running Task 18.1 (Threshold-Aware Adaptive Merging)"
echo "=================================================================="
python run_adaptive_merge.py \
    --base_model_dir "${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas" \
    --domain both \
    --target_ua_threshold 95.0 \
    --output_dir "${REPO_ROOT}/outputs/task1_merge" \
    --save_merged_ckpt

echo ""
echo "=================================================================="
echo " Step 3: Running Task 18.2 (Adaptive Anchor Assignment)"
echo "=================================================================="
python run_adaptive_anchor.py \
    --base_model_dir "${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas" \
    --domain both \
    --strategy all \
    --use_clip \
    --save_dir "${REPO_ROOT}/outputs/task2_anchor"

echo ""
echo "=================================================================="
echo " Step 4: Generating High-Resolution Figures in ./outputs/plots/"
echo "=================================================================="
python plot_results.py

echo ""
echo "=================================================================="
echo " All Tasks Completed Successfully!"
echo "   - Task 1 Metrics : outputs/task1_merge/task1_summary.json"
echo "   - Task 2 Metrics : outputs/task2_anchor/task2_summary.json"
echo "   - Figures        : outputs/plots/"
echo "=================================================================="
