#!/usr/bin/env bash
set -e

##############################################################################
# One‑shot Wan2.2 installation / launch script
#
# This script automates setup of the Wan2.2 environment and optionally
# downloads model weights via the HuggingFace CLI.  It is idempotent and can
# safely be run multiple times.  Provide an optional port as the first
# argument to override the default Gradio port.
##############################################################################

# ---- Configurable variables ----
WAN_PATH="${WAN_PATH:-$HOME/wan2.2}"
PORT="${1:-8888}"
PYTHON_CMD="${PYTHON_CMD:-python3.11}"
HF_REPO="Wan-AI/Wan2.2-TI2V-5B"

# 1) Prepare directory & Git repo
cd "$(dirname "$0")"
if [[ ! -d "$WAN_PATH" ]]; then
    echo "⟳  Cloning Wan2.2 repository …"
    git clone https://github.com/BenevolenceMessiah/Wan2.2.git "$WAN_PATH"
fi
cd "$WAN_PATH"
echo "⟳  git pull (update repository) …"
git pull || true

# 2) Python virtual environment
if [[ ! -d .venv ]]; then
    echo "⟳  Creating Python virtual environment …"
    "$PYTHON_CMD" -m venv .venv
fi
source .venv/bin/activate

# 3) System and build dependencies
echo "⟳  Installing build tools …"
sudo apt update -y
sudo apt install -y cmake build-essential ninja-build git-lfs
git lfs install || true

# 4) Install PyTorch with CUDA (customise CUDA version as needed)
echo "⟳  Installing PyTorch …"
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 5) Install Python requirements and optional flash_attn
echo "⟳  Installing Python dependencies …"
pip install -r requirements.txt || true
pip install --force-reinstall flash_attn --no-build-isolation || {
    echo "⚠️  flash_attn installation failed; continuing without it."
}

# 6) Download model weights via HuggingFace CLI (requires huggingface-cli)
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli is not installed.  Please run: pip install huggingface_hub"
    exit 1
fi
echo "⟳  Downloading model weights from HuggingFace …"
huggingface-cli download "$HF_REPO" --local-dir "Wan2.2-TI2V-5B" --repo-type model --resume-download || {
    echo "⚠️  Model download failed; continuing if files already exist."
}

# 7) Launch the web UI
echo "🚀  Starting Wan2.2 web UI on port $PORT …"
exec "$PYTHON_CMD" wan_web.py --port "$PORT"