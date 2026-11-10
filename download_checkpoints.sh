#!/bin/bash
set -euo pipefail

# ==============================================================================
# Download UnlearnCanvas Generator & Classifiers for CUIG (Task 1 & Task 2)
# Follows official instructions from Checkpoints/README.md
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Repository root: ${REPO_ROOT}"

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install -q gdown
fi

# 1. Download UnlearnCanvas Classifiers (style_classifier.pth, object_classifier.pth)
mkdir -p "${REPO_ROOT}/Checkpoints/Classifiers"
cd "${REPO_ROOT}/Checkpoints/Classifiers"

if [[ -f "UnlearnCanvas/style_classifier.pth" && -f "UnlearnCanvas/object_classifier.pth" ]]; then
    echo "[✓] UnlearnCanvas Classifiers already exist at Checkpoints/Classifiers/UnlearnCanvas/"
else
    echo "Downloading UnlearnCanvas Classifiers from Google Drive..."
    gdown --folder https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61
    rm -rf UnlearnCanvas
    mv cls_model UnlearnCanvas
    cd UnlearnCanvas
    rm -f style60.pth
    mv style50.pth style_classifier.pth
    mv style50_cls.pth object_classifier.pth
    echo "[✓] Classifiers ready at Checkpoints/Classifiers/UnlearnCanvas/"
fi

# 2. Download UnlearnCanvas Generator (Stable Diffusion fine-tuned on 50 styles & 20 objects)
mkdir -p "${REPO_ROOT}/Checkpoints/Generators"
cd "${REPO_ROOT}/Checkpoints/Generators"

if [[ -f "UnlearnCanvas/model_index.json" ]]; then
    echo "[✓] UnlearnCanvas Generator already exists at Checkpoints/Generators/UnlearnCanvas/"
else
    echo "Downloading UnlearnCanvas Generator from Google Drive..."
    gdown --folder https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx
    rm -rf UnlearnCanvas
    mv style50 UnlearnCanvas
    echo "[✓] Generator ready at Checkpoints/Generators/UnlearnCanvas/"
fi

cd "${REPO_ROOT}"
echo ""
echo "All official CUIG UnlearnCanvas checkpoints are ready!"
echo "  Generator : ${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas"
echo "  Classifier: ${REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas"
