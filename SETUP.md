# Setup: Klasifikasi Huruf ASL

Panduan menyiapkan environment dan menjalankan notebook `ASL_Klasifikasi_UAS_DL.ipynb`.

## Kebutuhan

- GPU NVIDIA dengan driver terbaru (cek dengan `nvidia-smi`). Diuji pada RTX 2060 6 GB.
- Python 3.11 atau 3.12.
- CUDA Toolkit tidak perlu dipasang terpisah.

## Instalasi

Windows (PowerShell):

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python check_env.py
```

Linux, macOS, atau Git Bash:

```bash
bash setup_env.sh
source .venv/bin/activate
python check_env.py
```

Skrip ini membuat `.venv`, memasang PyTorch dengan CUDA 12.6, memasang `requirements.txt`, dan mendaftarkan kernel Jupyter "Python (asl-env)". Untuk versi CUDA lain, pakai `.\setup_env.ps1 -Cuda <tag>` atau `CUDA=<tag> bash setup_env.sh`.

`check_env.py` akan menampilkan GPU dan menyarankan nilai `BATCH_SIZE`.

<details>
<summary>Instalasi manual</summary>

```bash
python -m venv .venv
# aktifkan .venv
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m ipykernel install --user --name asl-env --display-name "Python (asl-env)"
```

</details>

## Dataset

1. Unduh [American Sign Language Dataset](https://www.kaggle.com/datasets/iamavinashkr090502/asl-hand-landmark-and-gesture-dataset) (avinashkr09, CC BY-SA 4.0).
2. Ekstrak ke folder `data/` sehingga berisi subfolder `A/` sampai `Z/`.

Dataset tidak disertakan di repositori ini.

## Menjalankan

```bash
jupyter notebook
```

Buka `ASL_Klasifikasi_UAS_DL.ipynb`, pilih kernel "Python (asl-env)", lalu jalankan semua sel. Pengaturan ada di bagian 2 notebook:

| Variabel | Fungsi |
| --- | --- |
| `SMOKE_TEST` | `True` untuk uji cepat, `False` untuk pelatihan penuh |
| `LOCAL_DIR` | Folder dataset (default `./data`) |
| `BATCH_SIZE` | Ukuran batch (default 32) |

Hasil tersimpan di `outputs/`.

## Masalah umum

| Masalah | Solusi |
| --- | --- |
| PowerShell menolak menjalankan skrip | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `torch.cuda.is_available()` bernilai `False` | Pasang ulang torch dengan `--index-url` CUDA di atas, lalu perbarui driver |
| *Out of memory* | Turunkan `BATCH_SIZE` ke 16 atau 8 |
