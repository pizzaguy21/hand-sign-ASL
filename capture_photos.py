"""Ambil foto isyarat huruf A-Z dari webcam dan simpan per huruf.

Jalankan:
    python capture_photos.py                  # cari folder A..Z di ./data
    python capture_photos.py --target 40      # target 40 foto per huruf (mode otomatis)
    python capture_photos.py --camera 1       # jika webcam bawaan bukan indeks 0
    python capture_photos.py --out data/asl   # tentukan folder yang berisi A/ ... Z/
    python capture_photos.py --raw            # simpan foto polos tanpa kerangka

Tombol:
    SPASI      ambil satu foto
    A          nyalakan/matikan ambil otomatis (satu foto tiap ~0,8 detik sampai target tercapai)
    ENTER      huruf berikutnya
    BACKSPACE  huruf sebelumnya
    Q atau ESC keluar
Pratinjau dicerminkan agar mudah diarahkan, tetapi foto yang disimpan TIDAK dicerminkan.
"""
import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
SIZE = 200
GUIDE = (0, 200, 0)
RED, WHITE, MAGENTA = (0, 0, 255), (255, 255, 255), (255, 0, 255)
BORDER_PX = 3
MARGIN = 0.02                                       

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

def find_class_root(base: Path):
    """Cari folder yang berisi subfolder A..Z (boleh bersarang). Kembalikan None jika tidak ada."""
    if not base.exists():
        return None
    for d, subdirs, _ in os.walk(base):
        if {s.upper() for s in subdirs} >= set(LETTERS):
            return Path(d)
    return None


def letter_dir(root: Path, letter: str) -> Path:
    """Folder huruf yang sudah ada (huruf besar/kecil), atau buat baru."""
    for p in root.iterdir() if root.exists() else []:
        if p.is_dir() and p.name.upper() == letter:
            return p
    p = root / letter
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

def crop_box(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    return (w - s) // 2, (h - s) // 2, s


def to_crop_coords(landmarks, frame_shape):
    """Landmark ternormalisasi pada frame penuh -> ternormalisasi pada kotak persegi tengah."""
    h, w = frame_shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) / 2, (h - s) / 2
    return [((x * w - x0) / s, (y * h - y0) / s) for x, y in landmarks]


def hand_inside(crop_landmarks, margin=MARGIN):
    return all(margin <= x <= 1 - margin and margin <= y <= 1 - margin for x, y in crop_landmarks)


def draw_skeleton(img, landmarks, radius=3, thickness=2):
    """Gambar garis putih dan titik merah. landmarks: (x, y) ternormalisasi terhadap img."""
    h, w = img.shape[:2]
    pts = [(int(round(x * w)), int(round(y * h))) for x, y in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], WHITE, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, radius, RED, -1, cv2.LINE_AA)
    return img


def make_dataset_image(frame, crop_landmarks, raw):
    """Crop persegi tengah -> resize SIZE x SIZE -> (opsional) gambar kerangka. Tanpa flip."""
    x0, y0, s = crop_box(frame)
    img = cv2.resize(frame[y0:y0 + s, x0:x0 + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if not raw:
        draw_skeleton(img, crop_landmarks)
        if BORDER_PX > 0:
            draw_border(img, BORDER_PX)
    return img


def draw_border(img, px):
    """Bingkai merah muda (magenta) di tepi gambar, seperti gambar dataset Kaggle."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), MAGENTA, px * 2)
    return img


def save_photo(frame, crop_landmarks, folder: Path, letter: str, raw: bool) -> Path:
    path = folder / f"foto_{letter}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(path), make_dataset_image(frame, crop_landmarks, raw), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path

class HandDetector:
    """Membungkus MediaPipe Hands. Mencoba API 'solutions' lebih dulu, lalu API 'tasks'."""

    def __init__(self, model_path="hand_landmarker.task"):
        try:
            import mediapipe as mp
        except ImportError:
            raise SystemExit("MediaPipe belum terpasang. Jalankan:  pip install mediapipe")
        self.mp = mp
        self.kind = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.det = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.kind = "solutions"
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            path = Path(model_path)
            if not path.exists():
                print("Mengunduh model hand_landmarker.task (sekali saja) ...")
                urllib.request.urlretrieve(HAND_MODEL_URL, path)
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE, num_hands=1)
            self.det = vision.HandLandmarker.create_from_options(opts)
            self.kind = "tasks"
        print("MediaPipe Hands siap (API:", self.kind + ")")

    def detect(self, frame_bgr):
        """Kembalikan 21 titik [(x, y)] ternormalisasi pada frame penuh, atau None jika tidak ada tangan."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.kind == "solutions":
            res = self.det.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            return [(lm.x, lm.y) for lm in res.multi_hand_landmarks[0].landmark]
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.det.detect(mp_img)
        if not res.hand_landmarks:
            return None
        return [(lm.x, lm.y) for lm in res.hand_landmarks[0]]

    def close(self):
        if hasattr(self.det, "close"):
            self.det.close()

def selftest(root: Path):
    import numpy as np
    frame = (np.random.rand(480, 640, 3) * 100 + 100).astype("uint8")
    fake = [(0.40 + 0.01 * i, 0.30 + 0.015 * i) for i in range(21)]
    crop_lm = to_crop_coords(fake, frame.shape)
    assert hand_inside(crop_lm), "tangan palsu seharusnya berada di dalam kotak"
    assert not hand_inside(to_crop_coords([(0.01, 0.5)] * 21, frame.shape)), "tangan di tepi seharusnya ditolak"
    folder = letter_dir(root, "A")
    p = save_photo(frame, crop_lm, folder, "A", raw=False)
    img = cv2.imread(str(p))
    red = int(((img[:, :, 2] > 200) & (img[:, :, 1] < 80) & (img[:, :, 0] < 80)).sum())
    edge = img[0:2, :, :].reshape(-1, 3).mean(axis=0)
    border_ok = BORDER_PX == 0 or (edge[0] > 150 and edge[2] > 150 and edge[1] < 120)
    ok = img is not None and img.shape == (SIZE, SIZE, 3) and red > 20 and border_ok
    print("selftest:", "OK" if ok else "GAGAL", "| ukuran", img.shape, "| piksel merah:", red,
          "| bingkai magenta:", border_ok)
    p.unlink()
    return 0 if ok else 1

def draw_overlay(view, letter, saved, target, auto, found, msg):
    h, w = view.shape[:2]
    x0, y0, s = crop_box(view)
    cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), GUIDE, 2)
    cv2.putText(view, letter, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)
    status = f"{saved}/{target} foto" + ("  [OTOMATIS]" if auto else "")
    cv2.putText(view, status, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(view, "SPASI foto | A otomatis | ENTER next | BACKSPACE prev | Q keluar", (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    hand_txt, hand_col = ("TANGAN OK", (0, 200, 0)) if found else ("TANGAN TIDAK TERDETEKSI / TERPOTONG", (0, 0, 255))
    cv2.putText(view, hand_txt, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_col, 2)
    if msg:
        cv2.putText(view, msg, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="Ambil foto huruf ASL (dengan kerangka tangan) dari webcam.")
    ap.add_argument("--data", default="data", help="folder induk pencarian (default: data)")
    ap.add_argument("--out", default=None, help="folder yang berisi A/ ... Z/ (jika tidak diisi, dicari otomatis)")
    ap.add_argument("--camera", type=int, default=0, help="indeks webcam (default 0)")
    ap.add_argument("--target", type=int, default=30, help="target foto per huruf untuk mode otomatis (default 30)")
    ap.add_argument("--start", default="A", help="huruf awal (default A)")
    ap.add_argument("--delay", type=float, default=0.8, help="jeda mode otomatis, detik (default 0.8)")
    ap.add_argument("--raw", action="store_true", help="simpan foto polos tanpa kerangka (tetap butuh tangan terdeteksi)")
    ap.add_argument("--border", type=int, default=3, help="tebal bingkai merah muda dalam piksel (default 3, 0 = tanpa bingkai)")
    ap.add_argument("--model", default="hand_landmarker.task", help="lokasi model (hanya untuk API 'tasks')")
    ap.add_argument("--selftest", action="store_true", help="uji alur gambar tanpa webcam dan MediaPipe")
    args = ap.parse_args()
    global BORDER_PX
    BORDER_PX = max(0, args.border)

    if args.out:
        root = Path(args.out)
    else:
        root = find_class_root(Path(args.data))
        if root is None:
            root = Path(args.data)
            print(f"Folder A..Z belum ditemukan di '{args.data}'. Folder huruf baru akan dibuat di {root.resolve()}")
    root.mkdir(parents=True, exist_ok=True)
    print("Menyimpan ke:", root.resolve())

    if args.selftest:
        return selftest(root)

    detector = HandDetector(args.model)
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        print(f"Webcam indeks {args.camera} tidak bisa dibuka. Coba --camera 1, dan tutup aplikasi lain yang memakai kamera.")
        return 1

    idx = LETTERS.index(args.start.upper()) if args.start.upper() in LETTERS else 0
    auto, last_auto, msg, msg_until = False, 0.0, "", 0.0
    win = "Ambil foto huruf ASL"
    while True:
        letter = LETTERS[idx]
        folder = letter_dir(root, letter)
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca frame dari webcam.")
            break
        now = time.time()

        lm_full = detector.detect(frame)
        crop_lm = to_crop_coords(lm_full, frame.shape) if lm_full else None
        usable = crop_lm is not None and hand_inside(crop_lm)

        if auto and now - last_auto >= args.delay:
            if count_images(folder) >= args.target:
                auto, msg, msg_until = False, f"Target {args.target} foto tercapai. ENTER untuk huruf berikutnya.", now + 3
            elif usable:
                save_photo(frame, crop_lm, folder, letter, args.raw)
                last_auto = now

        preview = frame.copy()
        if lm_full:
            draw_skeleton(preview, lm_full, radius=4)
        view = cv2.flip(preview, 1)
        draw_overlay(view, letter, count_images(folder), args.target, auto, usable, msg if now < msg_until else "")
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == 32:
            if usable:
                save_photo(frame, crop_lm, folder, letter, args.raw)
                msg, msg_until = "Tersimpan", now + 0.5
            else:
                msg, msg_until = "Tidak disimpan: tangan belum terdeteksi penuh di kotak hijau", now + 1.5
        elif key in (ord("a"), ord("A")):
            auto, last_auto = not auto, now
        elif key in (13, 10):
            idx, auto = (idx + 1) % len(LETTERS), False
        elif key == 8:
            idx, auto = (idx - 1) % len(LETTERS), False

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("\nRingkasan jumlah foto per huruf:")
    print("  ".join(f"{L}:{count_images(letter_dir(root, L))}" for L in LETTERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
