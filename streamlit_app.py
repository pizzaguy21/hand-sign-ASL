"""Coba ResNet50 huruf ASL dari kamera HP lewat browser.

Pasang sekali:  pip install streamlit
Jalankan dari folder proyek (HTTPS diperlukan agar browser HP mengizinkan kamera):
    python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.sslCertFile certs/cert.pem --server.sslKeyFile certs/key.pem --server.fileWatcherType none --server.headless true --browser.gatherUsageStats false
Lalu buka https://<IP-laptop>:8501 di HP yang tersambung ke Wi-Fi yang sama.
Foto hanya diproses di laptop dan tidak disimpan.
"""
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

import demo_webcam as demo

ROOT = Path(__file__).resolve().parent
MAX_SIDE = 1280
TOP_K = 5
MOVING_LETTERS = {"J", "Z"}
WEIGHT_OPTIONS = {"utama": "Bobot utama (demo)",
                  "steven": "Uji orang baru: Steven",
                  "chela": "Uji orang baru: Chela",
                  "gea": "Uji orang baru: Gea"}


@st.cache_resource(show_spinner="Memuat deteksi tangan...")
def get_detector():
    return demo.HandDetector(str(ROOT / "hand_landmarker.task")), threading.Lock()


@st.cache_resource(show_spinner="Memuat model...")
def get_model(choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return demo.load_model(demo.weights_path(choice), device), device


def read_bgr(file, flip=False):
    """File gambar dari browser -> array BGR, mengikuti orientasi EXIF dari HP."""
    with Image.open(file) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    rgb.thumbnail((MAX_SIDE, MAX_SIDE))
    arr = np.asarray(rgb)
    if flip:
        arr = np.fliplr(arr).copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def analyze(bgr, model, device, detector, lock):
    """Kembalikan (pesan_masalah, probabilitas, gambar_untuk_model)."""
    with lock:
        landmarks = detector.detect(bgr)
    if landmarks is None:
        return "Tangan tidak terdeteksi. Coba tambah cahaya dan pakai latar yang polos.", None, None
    crop_landmarks = demo.to_crop_coords(landmarks, bgr.shape)
    if not demo.hand_inside(crop_landmarks):
        return "Tangan terpotong. Letakkan seluruh tangan di bagian tengah foto.", None, None
    image = demo.frame_to_model_image(bgr, crop_landmarks)
    return None, demo.predict_probs(model, image, device), image


def show_result(file, choice, flip):
    model, device = get_model(choice)
    detector, lock = get_detector()
    problem, probs, image = analyze(read_bgr(file, flip), model, device, detector, lock)
    if problem:
        st.warning(problem)
        return
    order = np.argsort(-probs)[:TOP_K]
    best = demo.LETTERS[order[0]]
    st.markdown(f"<div style='font-size:96px;font-weight:700;text-align:center;line-height:1.1'>{best}</div>", unsafe_allow_html=True)
    for k in order:
        st.progress(float(probs[k]), text=f"{demo.LETTERS[k]}: {100 * probs[k]:.1f}%")
    if best in MOVING_LETTERS:
        st.info(f"Huruf {best} aslinya bergerak, jadi hasil dari satu foto kurang dapat diandalkan.")
    st.caption("Persentase adalah keyakinan model, bukan akurasinya. Model bisa yakin tetapi salah.")
    st.image(image, caption="Gambar yang dilihat model", width=224)


def main():
    st.set_page_config(page_title="Demo Huruf ASL", page_icon="🤟", layout="centered")
    st.title("Demo Huruf ASL")
    st.caption("ResNet50, 26 huruf A–Z. Foto diproses di laptop dan tidak disimpan.")

    choice = st.selectbox("Bobot model", list(WEIGHT_OPTIONS), format_func=WEIGHT_OPTIONS.get)
    st.caption(demo.weights_label(choice) + ". Untuk uji yang jujur, pilih nama orang yang sedang mencoba.")
    flip = st.checkbox("Balik gambar kiri-kanan", help="Aktifkan bila foto dari HP tersimpan dalam keadaan tercermin.")

    camera_tab, upload_tab = st.tabs(["Kamera", "Unggah foto"])
    with camera_tab:
        shot = st.camera_input("Letakkan seluruh tangan di tengah, lalu ambil foto")
        if shot is not None:
            show_result(shot, choice, flip)
    with upload_tab:
        st.caption("Pakai ini bila kamera di atas tidak bisa dibuka.")
        uploaded = st.file_uploader("Ambil atau pilih foto", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            show_result(uploaded, choice, flip)


if __name__ == "__main__":
    main()
