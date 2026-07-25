#!/bin/bash
# =============================================================================
# Create the virtual environment for the RecSys Challenge 2026 pipeline.
#
# One environment covers everything: retrieval, reranker training, response
# generation, vLLM inference and the explanation-quality metrics.
#
# Usage:
#   bash setup_env.sh
#   PYTHON=python3.12 bash setup_env.sh     # pick the interpreter explicitly
#
# Then, in every session:
#   source .venv/bin/activate
#
# All commands in this repository are run from the repository root.
# =============================================================================
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "[1/4] Creating .venv/ ($("$PYTHON" --version))"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "[2/4] Installing dependencies (requirements.txt)"
# Resolved in one pass so pip sees vLLM's torch pin together with the rest.
pip install -r requirements.txt

echo "[3/4] Linking TorchCodec to pip-installed FFmpeg libraries"
patch-torchcodec --quiet

echo "[4/4] NLTK data required by the explanation-quality metrics"
# No '|| true' here: a silent NLTK failure only surfaces much later, inside the
# metrics, so fail now with an actionable message instead.
python - <<'PY'
import sys
import nltk
for pkg in ('punkt', 'punkt_tab'):
    nltk.download(pkg, quiet=True)
missing = []
for path in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
    try:
        nltk.data.find(path)
    except LookupError:
        missing.append(path)
if missing:
    sys.exit("NLTK data missing after download: " + ", ".join(missing) +
             "\nRetry with network access, or set NLTK_DATA to a directory that has it.")
print("nltk punkt + punkt_tab OK")
PY

echo "[preflight] Import checks"
python - <<'PY'
import sys
import torch, transformers
print(f"torch {torch.__version__} | transformers {transformers.__version__} | CUDA {torch.cuda.is_available()}")
from sentence_transformers import SentenceTransformer
print("sentence-transformers import OK")

# The Gemma generator and judge load a TimmWrapper vision tower. Without timm
# they fail at response-generation time, i.e. after retrieval and reranking have
# already run, so check it up front.
try:
    import timm
    from transformers.models.timm_wrapper import TimmWrapperModel  # noqa: F401
    print(f"timm {timm.__version__} + TimmWrapperModel OK (Gemma generator/judge)")
except Exception as e:
    sys.exit(f"Gemma models will not load: {type(e).__name__}: {e}\n"
             "Install the pinned timm from requirements.txt.")

try:
    import vllm; print(f"vllm {vllm.__version__}")
except Exception as e:
    print(f"vllm not importable: {e}")
PY

deactivate
echo "OK - environment ready: source .venv/bin/activate"
