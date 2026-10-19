#!/usr/bin/env python3
"""
generate_marker.py -- print-ready ArUco marker for the charging dock.

Produces the marker the docking module is configured to look for, at the
size it is configured to expect:

    dictionary  ARUCO_DICT_NAME       (config.py)
    id          ARUCO_MARKER_ID       (config.py)
    side        ARUCO_MARKER_SIZE_M   (config.py)

WHY SIZE ACCURACY MATTERS MORE THAN IT LOOKS
--------------------------------------------
solvePnP recovers range from the marker's apparent size against its known
physical size. Range error is proportional to size error: print at 95 mm
while the code believes 100 mm and every range reads 5 % short, all the
way to the contacts. A 100 mm marker printed "to fit" on A4 by a printer
driver that scales to 96 % puts the stopping point 6 mm out.

So this writes a PDF at exact physical scale, and prints a ruler beside
the marker so the scale can be checked with a steel rule before use. If
the ruler does not measure true, the print is wrong -- reprint it, do not
adjust the code.

    python3 generate_marker.py                      # PDF + PNG, defaults
    python3 generate_marker.py --id 0 --size-mm 100
    python3 generate_marker.py --verify out/marker_id0_100mm.png

Output goes to output/markers/.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ARUCO_DICT_NAME, ARUCO_MARKER_ID, ARUCO_MARKER_SIZE_M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output", "markers")
MM_PER_INCH = 25.4


def build_marker(dict_name: str, marker_id: int, px: int) -> np.ndarray:
    """Render the marker bitmap, crisply.

    generateImageMarker is given the exact pixel size so the module cells
    land on whole pixels; any rescale afterwards uses INTER_NEAREST for the
    same reason. A blurred marker edge costs corner-localisation accuracy,
    which is what the whole pose estimate rests on.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.generateImageMarker(dictionary, marker_id, px)


def verify(image: np.ndarray, dict_name: str, marker_id: int) -> bool:
    """Decode our own output. A marker that will not decode is worthless."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Quiet zone: the detector needs white around the marker to find it.
    padded = cv2.copyMakeBorder(gray, 40, 40, 40, 40,
                                cv2.BORDER_CONSTANT, value=255)
    corners, ids, _ = detector.detectMarkers(padded)
    if ids is None:
        print("  FAIL: no marker detected in the generated image")
        return False
    found = ids.ravel().tolist()
    if marker_id not in found:
        print(f"  FAIL: decoded {found}, expected {marker_id}")
        return False
    print(f"  decodes as id {marker_id} in {dict_name}")
    return True


def write_pdf(marker: np.ndarray, path: str, size_mm: float,
              marker_id: int, dict_name: str) -> None:
    """Write a PDF whose marker measures exactly `size_mm` on paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    page_w_mm, page_h_mm = 210.0, 297.0            # A4 portrait
    fig = plt.figure(figsize=(page_w_mm / MM_PER_INCH, page_h_mm / MM_PER_INCH))

    def ax_at(x_mm, y_mm, w_mm, h_mm):
        return fig.add_axes([x_mm / page_w_mm, y_mm / page_h_mm,
                             w_mm / page_w_mm, h_mm / page_h_mm])

    # Marker, placed by physical millimetres rather than by layout guesswork.
    mx = (page_w_mm - size_mm) / 2.0
    my = page_h_mm - size_mm - 45.0
    ax = ax_at(mx, my, size_mm, size_mm)
    ax.imshow(marker, cmap="gray", vmin=0, vmax=255,
              interpolation="none", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # A 100 mm ruler directly below, to check the printer did not rescale.
    ruler = ax_at(mx, my - 22.0, size_mm, 14.0)
    ruler.set_xlim(0, size_mm); ruler.set_ylim(0, 14)
    ruler.axis("off")
    ruler.add_patch(Rectangle((0, 6), size_mm, 0.6, color="black"))
    for mm in range(0, int(size_mm) + 1, 10):
        ruler.add_patch(Rectangle((mm - 0.15, 6), 0.3, 4.0, color="black"))
        ruler.text(mm, 11.5, str(mm), ha="center", va="bottom", fontsize=5)
    ruler.text(size_mm / 2.0, 0.5,
               f"This bar must measure exactly {size_mm:.0f} mm. "
               "If it does not, reprint at 100% scale.",
               ha="center", va="bottom", fontsize=6)

    fig.text(0.5, (my + size_mm + 18) / page_h_mm,
             "MediVan charging dock marker", ha="center", fontsize=13)
    fig.text(0.5, (my + size_mm + 11) / page_h_mm,
             f"{dict_name}   id {marker_id}   {size_mm:.0f} x {size_mm:.0f} mm",
             ha="center", fontsize=8)
    fig.text(0.5, 0.06,
             "Print at 100% / actual size -- NOT 'fit to page'.\n"
             "Matte paper. Mount flat on a rigid backing; a curled or\n"
             "creased marker biases the corner positions and the range with\n"
             "them. Keep the white border clear.",
             ha="center", fontsize=7, linespacing=1.6)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id", type=int, default=ARUCO_MARKER_ID)
    p.add_argument("--dict", default=ARUCO_DICT_NAME)
    p.add_argument("--size-mm", type=float, default=ARUCO_MARKER_SIZE_M * 1000.0)
    p.add_argument("--dpi", type=int, default=600,
                   help="PNG resolution; 600 dpi is ample for a 100 mm marker")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--verify", metavar="PNG",
                   help="decode an existing image instead of generating")
    args = p.parse_args()

    if args.verify:
        img = cv2.imread(args.verify, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"cannot read {args.verify}")
            return 1
        print(f"Verifying {args.verify}")
        return 0 if verify(img, args.dict, args.id) else 1

    px = int(round(args.size_mm / MM_PER_INCH * args.dpi))
    marker = build_marker(args.dict, args.id, px)

    stem = f"marker_id{args.id}_{int(args.size_mm)}mm"
    png = os.path.join(args.out_dir, stem + ".png")
    pdf = os.path.join(args.out_dir, stem + ".pdf")
    os.makedirs(args.out_dir, exist_ok=True)
    cv2.imwrite(png, marker)
    write_pdf(marker, pdf, args.size_mm, args.id, args.dict)

    print(f"Marker {args.dict} id {args.id}, {args.size_mm:.0f} mm")
    print(f"  PNG {png}  ({px}x{px} px at {args.dpi} dpi)")
    print(f"  PDF {pdf}  (A4, scale-exact -- print at 100%)")
    ok = verify(marker, args.dict, args.id)
    print("\nBefore using it: print the PDF, then measure the marker's black")
    print("outer edge with a steel rule. It must be "
          f"{args.size_mm:.0f} mm. ARUCO_MARKER_SIZE_M in config.py must")
    print("match what you actually printed, or every range will be wrong")
    print("by the same proportion.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
