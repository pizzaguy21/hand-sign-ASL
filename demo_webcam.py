"""Coba ResNet50 langsung dari webcam.

Jalankan:
    python demo_webcam.py                     # bobot utama (pernah melihat foto Steven, Chela, Gea)
    python demo_webcam.py --weights steven    # uji orang baru: model tidak pernah melihat foto Steven
    python demo_webcam.py --weights chela
    python demo_webcam.py --weights gea
    python demo_webcam.py --camera 1          # jika webcam bawaan bukan indeks 0

Tombol: Q atau ESC keluar.
Pratinjau dicerminkan agar mudah diarahkan, tetapi gambar untuk model TIDAK dicerminkan, sama seperti data latih.
"""
import argparse
import collections
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from scipy import ndimage
from torchvision import models
from torchvision import transforms as T

from capture_photos import (LETTERS, HandDetector, crop_box, draw_skeleton, hand_inside,
                            make_dataset_image, to_crop_coords)

IMG_SIZE, CANVAS_SIZE, HAND_MARGIN = 224, 300, 0.13
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OUTPUTS = Path(__file__).resolve().parent / "outputs" / "full"
PEOPLE = ["steven", "chela", "gea"]
TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def weights_path(choice):
    folder = OUTPUTS if choice == "utama" else OUTPUTS / "holdout" / choice
    return folder / "checkpoints" / "resnet50_best.pt"


def weights_label(choice):
    if choice == "utama":
        return "Bobot utama: pernah melihat foto Steven, Chela, Gea"
    return f"Uji orang baru: tidak pernah melihat foto {choice.capitalize()}"


def load_model(path, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LETTERS))
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def find_hand_box(rgb, border=6):
    """Kotak pembatas tangan dari titik sendi merah yang sudah tergambar. None jika titik tidak terbaca."""
    a = np.asarray(rgb).astype(np.int16)
    mask = (a[..., 0] > 170) & (a[..., 1] < 90) & (a[..., 2] < 90)
    mask[:border, :] = mask[-border:, :] = False
    mask[:, :border] = mask[:, -border:] = False
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    ids = np.arange(1, n + 1)
    sizes = ndimage.sum_labels(mask, labels, index=ids)
    keep = ids[(sizes >= 4) & (sizes <= 1000)]
    if len(keep) == 0 or sizes[keep - 1].sum() < 60:
        return None
    ys, xs = np.nonzero(np.isin(labels, keep))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def standardize_photo(rgb):
    """Samakan dengan format Kaggle: crop area tangan, bingkai tipis, di tengah kanvas putih persegi."""
    W, H = rgb.size
    edge = 5
    box = find_hand_box(rgb)
    if box is None:
        x0, y0, x1, y1 = edge, edge, W - 1 - edge, H - 1 - edge
    else:
        x0, y0, x1, y1 = box
        m = int(round(HAND_MARGIN * max(x1 - x0, y1 - y0)))
        x0, y0 = max(edge, x0 - m), max(edge, y0 - m)
        x1, y1 = min(W - 1 - edge, x1 + m), min(H - 1 - edge, y1 + m)
    crop = rgb.crop((x0, y0, x1 + 1, y1 + 1))
    scale = CANVAS_SIZE / max(crop.size)
    nw, nh = max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale))
    crop = crop.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    px, py = (CANVAS_SIZE - nw) // 2, (CANVAS_SIZE - nh) // 2
    canvas.paste(crop, (px, py))
    ImageDraw.Draw(canvas).rectangle([px, py, px + nw - 1, py + nh - 1], outline=(255, 0, 255), width=3)
    return canvas.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def frame_to_model_image(frame, crop_landmarks):
    """Frame webcam -> gambar 224x224 dengan alur yang sama seperti foto kelompok saat pelatihan."""
    bgr = make_dataset_image(frame, crop_landmarks, raw=False)
    return standardize_photo(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


@torch.inference_mode()
def predict_probs(model, image, device):
    return torch.softmax(model(TO_TENSOR(image).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()


def draw_overlay(view, choice, probs, usable, fps):
    h, w = view.shape[:2]
    x0, y0, s = crop_box(view)
    cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), (0, 200, 0), 2)
    cv2.putText(view, weights_label(choice), (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(view, f"Q keluar | {fps:.0f} FPS", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if probs is None or not usable:
        cv2.putText(view, "TANGAN TIDAK TERDETEKSI / TERPOTONG", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    top = np.argsort(-probs)[:3]
    cv2.putText(view, LETTERS[top[0]], (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 255), 7)
    for i, k in enumerate(top):
        cv2.putText(view, f"{LETTERS[k]}  {100 * probs[k]:.0f}%", (15, 150 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def paste_model_view(view, image, size=160):
    """Tampilkan gambar yang benar-benar dilihat model di pojok kanan atas."""
    inset = cv2.cvtColor(np.asarray(image.resize((size, size))), cv2.COLOR_RGB2BGR)
    view[10:10 + size, view.shape[1] - size - 10:view.shape[1] - 10] = inset


def main():
    ap = argparse.ArgumentParser(description="Coba ResNet50 huruf ASL langsung dari webcam.")
    ap.add_argument("--weights", choices=["utama"] + PEOPLE, default="utama",
                    help="utama = bobot utama; steven/chela/gea = bobot uji orang baru tanpa foto orang itu")
    ap.add_argument("--camera", type=int, default=0, help="indeks webcam (default 0)")
    ap.add_argument("--smooth", type=int, default=10, help="jumlah frame yang dirata-ratakan (default 10)")
    ap.add_argument("--model", default="hand_landmarker.task", help="lokasi model MediaPipe (hanya untuk API 'tasks')")
    args = ap.parse_args()

    path = weights_path(args.weights)
    if not path.exists():
        print("Bobot tidak ditemukan:", path)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(path, device)
    print("Bobot:", path, "| perangkat:", device)
    print(weights_label(args.weights))

    detector = HandDetector(args.model)
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        print(f"Webcam indeks {args.camera} tidak bisa dibuka. Coba --camera 1, dan tutup aplikasi lain yang memakai kamera.")
        detector.close()
        return 1

    recent = collections.deque(maxlen=max(1, args.smooth))
    last, fps = time.time(), 0.0
    win = "Demo huruf ASL - ResNet50"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca frame dari webcam.")
            break
        lm_full = detector.detect(frame)
        crop_lm = to_crop_coords(lm_full, frame.shape) if lm_full else None
        usable = crop_lm is not None and hand_inside(crop_lm)

        image = None
        if usable:
            image = frame_to_model_image(frame, crop_lm)
            recent.append(predict_probs(model, image, device))
        else:
            recent.clear()

        preview = frame.copy()
        if lm_full:
            draw_skeleton(preview, lm_full, radius=4)
        view = cv2.flip(preview, 1)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        draw_overlay(view, args.weights, np.mean(recent, axis=0) if recent else None, usable, fps)
        if image is not None:
            paste_model_view(view, image)
        cv2.imshow(win, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
