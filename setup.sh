#!/bin/bash
# =====================================================================
# Brev Launchable setup — Module 3: Generative AI & LLMs live demo
# Paste this into the "Do you want to run a setup script?" step of the
# Brev Launchable wizard (VM Mode w/ Jupyter).
# =====================================================================
set -euo pipefail

echo ">>> [1/3] Python packages"
pip install --upgrade pip
# torch is already present in NVIDIA PyTorch containers; install only if missing
python -c "import torch" 2>/dev/null || pip install torch
pip install "transformers>=4.44" gensim bertviz sentencepiece ipywidgets scipy "numpy<2"

echo ">>> [2/3] Pre-cache models & embeddings (so class time has NO downloads)"
python - <<'PY'
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM)
# GPT-2 (Cells 2 & 3)
AutoTokenizer.from_pretrained("gpt2")
AutoModelForCausalLM.from_pretrained("gpt2")
AutoModel.from_pretrained("gpt2", output_attentions=True)
# FLAN-T5 (Cell 4)
AutoTokenizer.from_pretrained("google/flan-t5-base")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
# GloVe embeddings (Cell 1)
import gensim.downloader as api
api.load("glove-wiki-gigaword-100")
print("All models & embeddings cached.")
PY

echo ">>> [3/3] Done. Open Mod3_demo.ipynb in JupyterLab and run top to bottom."
