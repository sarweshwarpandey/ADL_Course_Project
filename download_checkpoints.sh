#!/bin/bash
set -euo pipefail

# ==============================================================================
# Download UnlearnCanvas Generator & Classifiers for CUIG (Task 1 & Task 2)
# Uses Hugging Face Hub (`NormalUhr/UnlearnCanvas-ckpts`) to avoid Google Drive
# `gdown --folder` rate limits / "Failed to retrieve folder contents" errors.
# Matches exact layout expected by Checkpoints/README.md:
#   - Checkpoints/Classifiers/UnlearnCanvas/style_classifier.pth
#   - Checkpoints/Classifiers/UnlearnCanvas/object_classifier.pth
#   - Checkpoints/Generators/UnlearnCanvas/ (Diffusers pipeline)
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Repository root: ${REPO_ROOT}"

python - "${REPO_ROOT}" << 'EOF'
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

repo_root = Path(sys.argv[1])
hf_repo = "NormalUhr/UnlearnCanvas-ckpts"

# 1. Download Classifiers
cls_dir = repo_root / "Checkpoints" / "Classifiers" / "UnlearnCanvas"
cls_dir.mkdir(parents=True, exist_ok=True)

style_cls_target = cls_dir / "style_classifier.pth"
obj_cls_target = cls_dir / "object_classifier.pth"

if style_cls_target.exists() and obj_cls_target.exists():
    print("[✓] UnlearnCanvas Classifiers already exist at Checkpoints/Classifiers/UnlearnCanvas/")
else:
    print("Downloading UnlearnCanvas Classifiers from HuggingFace (NormalUhr/UnlearnCanvas-ckpts)...")
    style_path = hf_hub_download(repo_id=hf_repo, filename="cls_model/style50.pth")
    shutil.copy2(style_path, style_cls_target)
    print(f"  -> Saved {style_cls_target}")

    obj_path = hf_hub_download(repo_id=hf_repo, filename="cls_model/style50_cls.pth")
    shutil.copy2(obj_path, obj_cls_target)
    print(f"  -> Saved {obj_cls_target}")
    print("[✓] Classifiers ready at Checkpoints/Classifiers/UnlearnCanvas/")

# 2. Download UnlearnCanvas Generator (diffusion/diffusers/style50)
gen_dir = repo_root / "Checkpoints" / "Generators" / "UnlearnCanvas"
if (gen_dir / "model_index.json").exists():
    print("[✓] UnlearnCanvas Generator already exists at Checkpoints/Generators/UnlearnCanvas/")
else:
    print("Downloading UnlearnCanvas Generator from HuggingFace (NormalUhr/UnlearnCanvas-ckpts)...")
    tmp_dl = repo_root / "Checkpoints" / "Generators" / "_hf_tmp"
    snapshot_download(
        repo_id=hf_repo,
        allow_patterns=["diffusion/diffusers/style50/*"],
        local_dir=str(tmp_dl),
    )
    extracted_dir = tmp_dl / "diffusion" / "diffusers" / "style50"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    shutil.move(str(extracted_dir), str(gen_dir))
    shutil.rmtree(tmp_dl, ignore_errors=True)
    print("[✓] Generator ready at Checkpoints/Generators/UnlearnCanvas/")
EOF

echo ""
echo "All official CUIG UnlearnCanvas checkpoints are ready!"
echo "  Generator : ${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas"
echo "  Classifier: ${REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas"
