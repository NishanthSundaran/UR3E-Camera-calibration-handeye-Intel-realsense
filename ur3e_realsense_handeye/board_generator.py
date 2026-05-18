#!/usr/bin/env python3
"""Generate a printable ChArUco board PNG that matches the calibration node.

Usage:
  ros2 run ur3e_realsense_handeye board_generator \\
      --squares_x 8 --squares_y 11 \\
      --square_length_mm 29 --marker_length_mm 21 \\
      --dpi 300 --out charuco_8x11.png

If you change ``squares_x``, ``squares_y``, ``square_length_mm`` or
``marker_length_mm`` here, pass the same values to the calibration node
(``--ros-args -p squares_x:=... -p square_length_m:=...``) so the board
geometry matches.

Print at the configured DPI and **measure the actual printed squares with
a ruler before mounting** — printer scaling can introduce a few percent
of error, which directly biases the calibration result.
"""

from __future__ import annotations
import argparse
import os
import sys

import cv2
import numpy as np


def generate(
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dpi: int,
    out_path: str,
    margin_mm: float = 10.0,
) -> str:
    """Render the board and write it to ``out_path``."""
    if marker_length_mm >= square_length_mm:
        raise ValueError(
            "marker_length_mm must be strictly smaller than square_length_mm")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm / 1000.0,
        marker_length_mm / 1000.0,
        dictionary,
    )

    # Resolution: dpi pixels per inch -> px per mm.
    px_per_mm = dpi / 25.4
    img_w_px = int(round((squares_x * square_length_mm + 2 * margin_mm) * px_per_mm))
    img_h_px = int(round((squares_y * square_length_mm + 2 * margin_mm) * px_per_mm))
    margin_px = int(round(margin_mm * px_per_mm))

    img = board.generateImage(
        (img_w_px, img_h_px),
        marginSize=margin_px,
        borderBits=1,
    )

    out_path = os.path.abspath(out_path)
    cv2.imwrite(out_path, img)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a ChArUco calibration board PNG.")
    parser.add_argument("--squares_x", type=int, default=8)
    parser.add_argument("--squares_y", type=int, default=11)
    parser.add_argument("--square_length_mm", type=float, default=29.0)
    parser.add_argument("--marker_length_mm", type=float, default=21.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--margin_mm", type=float, default=10.0)
    parser.add_argument("--out", type=str, default="charuco_board.png")
    args = parser.parse_args()

    path = generate(
        args.squares_x, args.squares_y,
        args.square_length_mm, args.marker_length_mm,
        args.dpi, args.out, args.margin_mm,
    )
    width_mm = args.squares_x * args.square_length_mm
    height_mm = args.squares_y * args.square_length_mm
    print(f"Wrote {path}")
    print(f"  Grid:   {args.squares_x} x {args.squares_y} squares")
    print(f"  Square: {args.square_length_mm:.1f} mm  (marker {args.marker_length_mm:.1f} mm)")
    print(f"  Sheet:  {width_mm:.1f} x {height_mm:.1f} mm at {args.dpi} dpi")
    print("  Print at 100% scale (no fit-to-page) and verify with a ruler.")


if __name__ == "__main__":
    sys.exit(main())
