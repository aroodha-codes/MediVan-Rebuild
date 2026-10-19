"""
frame_source.py -- live camera backends for the physical robot.

One job: hand back a BGR uint8 frame at FRAME_W x FRAME_H, or None.

Two backends are tried in order:

    1. Picamera2        -- Raspberry Pi CSI camera via libcamera. This is
                           the one that works on a Pi Camera Module V1
                           (OV5647) on current Raspberry Pi OS.
    2. cv2.VideoCapture -- USB webcams, and laptops during development.

WHY Picamera2 IS TRIED FIRST
----------------------------
On Raspberry Pi OS Bookworm the CSI camera is driven by libcamera.
`cv2.VideoCapture(0)` goes through V4L2/GStreamer and typically fails with

    v4l2src0 reported: Failed to allocate required memory
    unable to start pipeline

The dangerous part is that `cap.isOpened()` still returns True in that
state, so code that trusts it reports a working camera while every read
returns None.

Nothing here trusts isOpened(). A backend is only accepted after it has
actually produced a frame.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Optional

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FRAME_W, FRAME_H, FPS

#: Force a backend when diagnosing: "picamera2", "opencv".
_FORCE_BACKEND = os.environ.get("MEDIVAN_CAMERA", "").strip().lower()

#: Picamera2 channel order override. See _Picamera2Source.
_PICAM_SWAP_RB = os.environ.get("MEDIVAN_PICAM_SWAP_RB", "0") == "1"

#: Seconds to let auto-exposure and auto-white-balance settle. Frames
#: grabbed before this are dark, and a dark frame makes floor
#: segmentation look broken when it is not.
_WARMUP_S = float(os.environ.get("MEDIVAN_CAMERA_WARMUP", "2.0"))


class FrameSource:
    """Interface: a name, a read(), a release()."""

    name = "none"

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class _Picamera2Source(FrameSource):
    """Raspberry Pi CSI camera (OV5647 / IMX219) via libcamera.

    COLOUR ORDER
    ------------
    Picamera2 format names describe the packed 32-bit word, not the byte
    order of the numpy array, so they read backwards: requesting
    "RGB888" returns an array whose channels are already B, G, R -- i.e.
    OpenCV-native BGR, needing no conversion.

    That is the documented behaviour of the standard libcamera stack, and
    it is what this class assumes. It has NOT been verified against your
    specific sensor here. If colours are inverted (a red object reading
    as blue), run everything with:

        MEDIVAN_PICAM_SWAP_RB=1

    Check it once with `python3 hardware/frame_source.py --check`. Wrong
    channel order degrades YOLO accuracy quietly rather than obviously.
    """

    name = "picamera2"

    def __init__(self) -> None:
        from picamera2 import Picamera2 # type: ignore

        self._picam = Picamera2()
        config = self._picam.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            buffer_count=4,
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(_WARMUP_S)
        self._closed = False

    def read(self) -> Optional[np.ndarray]:
        if self._closed:
            return None
        try:
            frame = self._picam.capture_array()
        except Exception:                        # noqa: BLE001
            return None
        if frame is None:
            return None

        # Some configurations deliver XRGB8888 (4 channels). Drop alpha
        # here rather than letting a 4-channel array reach OpenCV, where
        # it fails somewhere much less obvious.
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if _PICAM_SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        return np.ascontiguousarray(frame)

    def release(self) -> None:
        """Stop then close. Skipping close() leaves the sensor claimed and
        the next process gets 'Device or resource busy'."""
        if self._closed:
            return
        self._closed = True
        try:
            self._picam.stop()
        except Exception:                        # noqa: BLE001
            pass
        try:
            self._picam.close()
        except Exception:                        # noqa: BLE001
            pass


class _VideoCaptureSource(FrameSource):
    """USB webcam via OpenCV. Platform-correct backend flag."""

    name = "opencv"

    def __init__(self, index: int = 0) -> None:
        system = platform.system()
        if system == "Windows":
            backend = cv2.CAP_DSHOW
        elif system == "Linux":
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY

        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"camera index {index} would not open")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        # isOpened() is not proof. Demand an actual frame.
        ok, test = cap.read()
        if not ok or test is None:
            cap.release()
            raise RuntimeError(f"camera index {index} opened but read nothing")

        self._cap = cap

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def open_frame_source() -> Optional[FrameSource]:
    """Return the first backend that actually delivers a frame, or None.

    Every rejection prints why. A camera silently falling back is the
    failure this module exists to prevent.
    """
    candidates = []
    if _FORCE_BACKEND in ("", "picamera2"):
        candidates.append(("Picamera2", _Picamera2Source))
    if _FORCE_BACKEND in ("", "opencv"):
        candidates.append(("OpenCV VideoCapture", _VideoCaptureSource))

    for label, factory in candidates:
        try:
            source = factory()
        except ImportError:
            print(f"[Camera] {label} not installed")
            continue
        except Exception as exc:                 # noqa: BLE001
            print(f"[Camera] {label} unavailable: {exc}")
            continue

        probe = source.read()
        if probe is None:
            print(f"[Camera] {label} opened but returned no frame")
            source.release()
            continue

        print(f"[Camera] LIVE via {label} -- "
              f"{probe.shape[1]}x{probe.shape[0]} BGR")
        return source

    return None


def _check() -> int:
    """python3 hardware/frame_source.py --check"""
    src = open_frame_source()
    if src is None:
        print("\nNo camera. On a Pi check, in order:")
        print("  rpicam-hello --list-cameras")
        print("  python3 -c 'from picamera2 import Picamera2'")
        print("  venv must be created with --system-site-packages")
        return 1

    t0, n, frame = time.perf_counter(), 0, None
    for _ in range(30):
        f = src.read()
        if f is not None:
            frame, n = f, n + 1
    dt = time.perf_counter() - t0

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "output")
    os.makedirs(out, exist_ok=True)
    path = os.path.abspath(os.path.join(out, "camera_check.png"))
    cv2.imwrite(path, frame)

    b, g, r = (float(frame[:, :, i].mean()) for i in range(3))
    print(f"\n  backend       {src.name}")
    print(f"  frames        {n}/30 in {dt:.2f}s = {n/dt:.1f} fps")
    print(f"  brightness    {frame.mean():.1f}/255"
          + ("   TOO DARK -- add light" if frame.mean() < 60 else ""))
    print(f"  channels      B={b:.1f} G={g:.1f} R={r:.1f}")
    print(f"  saved         {path}")
    print("\n  Point the camera at something strongly RED and re-run.")
    print("  If it looks BLUE in that image, re-run everything with")
    print("  MEDIVAN_PICAM_SWAP_RB=1")
    src.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(_check())