#!/usr/bin/env bash
# Setup environment (Linux / macOS / Git Bash).
# Jalankan dari folder proyek:   bash setup_env.sh
# Opsi:  CUDA=cu126 PYTHON=python3.12 bash setup_env.sh   (tag CUDA sesuai perintah terbaru di pytorch.org)
set -euo pipefail

CUDA="${CUDA:-cu126}"
PYTHON="${PYTHON:-python3}"

echo "1/5 Membuat virtual environment .venv ..."
"$PYTHON" -m venv .venv

if [ -f .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi

echo "2/5 Memperbarui pip ..."
"$VPY" -m pip install --upgrade pip

echo "3/5 Memasang PyTorch (CUDA: $CUDA) ..."
"$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"

echo "4/5 Memasang paket proyek ..."
"$VPY" -m pip install -r requirements.txt

echo "5/5 Mendaftarkan kernel Jupyter dan membuat folder ..."
"$VPY" -m ipykernel install --user --name asl-env --display-name "Python (asl-env)"
mkdir -p data outputs notebooks

echo
echo "Selesai. Aktifkan environment dengan:  source .venv/bin/activate   (Windows Git Bash: source .venv/Scripts/activate)"
echo "Lalu periksa GPU dengan:               python check_env.py"
