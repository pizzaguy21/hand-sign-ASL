# Setup environment (Windows PowerShell).
# Jalankan dari folder proyek:   .\setup_env.ps1
# Opsi:  .\setup_env.ps1 -Cuda cu126      (ganti tag CUDA sesuai perintah terbaru di pytorch.org)
param(
    [string]$Cuda = "cu126",
    [string]$Python = "py -3.12"
)

$ErrorActionPreference = "Stop"

Write-Host "1/5 Membuat virtual environment .venv ..."
Invoke-Expression "$Python -m venv .venv"

Write-Host "2/5 Mengaktifkan dan memperbarui pip ..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "3/5 Memasang PyTorch (CUDA: $Cuda) ..."
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"

Write-Host "4/5 Memasang paket proyek ..."
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "5/5 Mendaftarkan kernel Jupyter dan membuat folder ..."
& .\.venv\Scripts\python.exe -m ipykernel install --user --name asl-env --display-name "Python (asl-env)"
New-Item -ItemType Directory -Force -Path data, outputs, notebooks | Out-Null

Write-Host ""
Write-Host "Selesai. Aktifkan environment dengan:  .\.venv\Scripts\Activate.ps1"
Write-Host "Lalu periksa GPU dengan:               python check_env.py"
