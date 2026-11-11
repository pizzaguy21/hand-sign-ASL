import importlib
import platform
import sys

OK, BAD = "[OK]  ", "[GAGAL]"


def check_packages():
    print("== Paket ==")
    missing = []
    for name in ["numpy", "pandas", "matplotlib", "sklearn", "PIL", "tqdm", "torch", "torchvision"]:
        try:
            mod = importlib.import_module(name)
            print(f"{OK} {name} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"{BAD} {name} belum terpasang")
            missing.append(name)
    return missing


def check_gpu():
    import torch
    print("\n== GPU ==")
    print(f"torch {torch.__version__} | CUDA runtime torch: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print(f"{BAD} CUDA tidak tersedia. Kemungkinan: torch versi CPU, atau driver NVIDIA belum terpasang/terlalu lama.")
        print("       Pasang ulang torch dengan indeks CUDA (lihat SETUP.md) dan perbarui driver NVIDIA.")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"{OK} GPU: {props.name} | VRAM {props.total_memory / 1024**3:.1f} GB | "
          f"compute capability {props.major}.{props.minor}")
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print(f"{OK} operasi matriks di GPU berjalan (hasil {float((x @ x).mean()):.4f})")
    return True


def check_batch_size():
    import torch
    import torchvision
    print("\n== Uji memori: ResNet50, 224x224, mixed precision ==")
    model = torchvision.models.resnet50(weights=None, num_classes=26).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    fits = []
    for bs in (16, 32, 64):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 3, 224, 224, device="cuda")
            y = torch.randint(0, 26, (bs,), device="cuda")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"{OK} batch {bs}: muat, memori puncak {peak:.2f} GB")
            fits.append(bs)
        except torch.cuda.OutOfMemoryError:
            print(f"{BAD} batch {bs}: kehabisan memori GPU")
            break
    if fits:
        print(f"-> Gunakan BATCH_SIZE <= {max(b for b in fits if b <= 32)} pada notebook.")
    else:
        print("-> Gunakan BATCH_SIZE = 8 atau kurang, dan tutup aplikasi lain yang memakai GPU.")


def main():
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
    major, minor = sys.version_info[:2]
    if not (3, 10) <= (major, minor) <= (3, 13):
        print(f"{BAD} Disarankan Python 3.10 sampai 3.13 (3.11 atau 3.12 paling aman).")
    missing = check_packages()
    if "torch" in missing or "torchvision" in missing:
        print("\nPyTorch belum terpasang. Jalankan setup_env terlebih dahulu (lihat SETUP.md).")
        return 1
    if check_gpu():
        check_batch_size()
    return 0


if __name__ == "__main__":
    sys.exit(main())
