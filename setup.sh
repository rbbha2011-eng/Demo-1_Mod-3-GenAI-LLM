#!/bin/bash
# =====================================================================
# Brev setup — Module 3: Generative AI & LLMs live demo
# Tested on an NVIDIA L40S instance (VM Mode w/ Jupyter).
#
# Run this once in a JupyterLab Terminal after cloning the repo:
#     cd Demo-1_Mod-3-GenAI-LLM
#     bash setup.sh
#
# It handles the two gotchas found in testing:
#   1) The instance's .venv has no "pip" -> we bootstrap it with ensurepip.
#   2) Newest transformers/torch don't match the CUDA-12 GPU driver
#      -> we pin transformers 4.44.2 and torch 2.4.1 (cu121).
# =====================================================================
set -euo pipefail

echo ">>> [1/4] Bootstrap pip inside the environment"
python3 -m ensurepip --upgrade || curl -sS https://bootstrap.pypa.io/get-pip.py | python3
python3 -m pip install --upgrade pip

echo ">>> [2/4] Install pinned packages (versions that work together)"
python3 -m pip install \
  "transformers==4.44.2" \
  gensim bertviz sentencepiece ipywidgets scipy matplotlib "numpy<2"

echo ">>> [3/4] Install the CUDA-12 build of PyTorch (matches the GPU driver)"
python3 -m pip install --force-reinstall "torch==2.4.1" \
  --index-url https://download.pytorch.org/whl/cu121

echo ">>> [4/4] Pre-cache models & embeddings (so class time has NO downloads)"
python3 - <<'PY'
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM)
AutoTokenizer.from_pretrained("gpt2")
AutoModelForCausalLM.from_pretrained("gpt2")
AutoModel.from_pretrained("gpt2", output_attentions=True, attn_implementation="eager")
AutoTokenizer.from_pretrained("google/flan-t5-base")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
import gensim.downloader as api
api.load("glove-wiki-gigaword-100")
print("All models & embeddings cached.")
PY

echo ">>> Verifying GPU is visible to PyTorch:"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo ">>> Done. Open Mod3_demo.ipynb in JupyterLab and run top to bottom."
