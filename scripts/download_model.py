"""
download_model.py — Download and prepare YOLOv8n ONNX model for MediVan.

Attempts three strategies in order:
  1. Direct download from Ultralytics GitHub releases
  2. pip install ultralytics + yolo export
  3. Print manual instructions

Usage:
    python scripts/download_model.py
"""

import os
import sys
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ONNX_PATH = os.path.join(ASSETS_DIR, "yolov8n.onnx")

# Direct download URL (Ultralytics official)
ONNX_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"


def download_direct():
    """Try downloading pre-exported ONNX from GitHub releases."""
    print("[download] Attempting direct download from Ultralytics GitHub...")
    try:
        import urllib.request
        os.makedirs(ASSETS_DIR, exist_ok=True)

        # Download with progress
        def report(block, block_size, total):
            pct = block * block_size / max(total, 1) * 100
            print(f"\r  Downloading: {pct:.0f}%", end="", flush=True)

        urllib.request.urlretrieve(ONNX_URL, ONNX_PATH, reporthook=report)
        print()

        size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        if size_mb < 1.0:
            print(f"[download] File too small ({size_mb:.1f} MB) — likely invalid.")
            os.remove(ONNX_PATH)
            return False

        print(f"[download] Success! Model saved -> {ONNX_PATH} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"[download] Direct download failed: {e}")
        return False


def export_via_ultralytics():
    """Try using the ultralytics package to export."""
    print("\n[download] Attempting export via ultralytics package...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("  ultralytics installed.")
    except Exception:
        print("  Could not install ultralytics.")
        return False

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        export_path = model.export(format="onnx", imgsz=320)
        if export_path and os.path.exists(export_path):
            os.makedirs(ASSETS_DIR, exist_ok=True)
            shutil.move(str(export_path), ONNX_PATH)
            size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
            print(f"[download] Export success! {ONNX_PATH} ({size_mb:.1f} MB)")
            return True
    except Exception as e:
        print(f"[download] Export failed: {e}")
    return False


def validate_model():
    """Quick sanity check: load ONNX with OpenCV DNN."""
    print("\n[download] Validating model with OpenCV DNN...")
    try:
        import cv2
        net = cv2.dnn.readNetFromONNX(ONNX_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Dummy forward pass
        import numpy as np
        blob = np.zeros((1, 3, 320, 320), dtype=np.float32)
        net.setInput(blob)
        output = net.forward()
        print(f"  Output shape: {output.shape}")
        print("  ✅ Model validated successfully!")
        return True
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False


def print_manual_instructions():
    """Print manual download steps."""
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"""
  Option A — Download pre-exported ONNX:
    URL: {ONNX_URL}
    Save to: {ONNX_PATH}

  Option B — Export from PyTorch:
    pip install ultralytics
    yolo export model=yolov8n.pt format=onnx imgsz=320
    Move yolov8n.onnx -> {ONNX_PATH}

  The simulator works without ONNX (uses heuristic classifier).
  ONNX is only needed for real camera deployment.
""")


def main():
    print("=" * 60)
    print("  MediVan — YOLOv8n ONNX Model Downloader")
    print("=" * 60)

    if os.path.exists(ONNX_PATH):
        size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        print(f"\n  Model already exists: {ONNX_PATH} ({size_mb:.1f} MB)")
        if validate_model():
            print("\n  No action needed — model is ready!")
            return
        print("  Existing model is invalid, re-downloading...")

    if download_direct():
        validate_model()
        return

    if export_via_ultralytics():
        validate_model()
        return

    print_manual_instructions()


if __name__ == "__main__":
    main()
