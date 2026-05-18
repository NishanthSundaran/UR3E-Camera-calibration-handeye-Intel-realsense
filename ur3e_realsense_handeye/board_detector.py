"""ChArUco / ArUco board detection and pose estimation.

The detector runs four passes over each frame:
  1. ChArUco corner detection in the configured (sx, sy) orientation.
  2. ChArUco corner detection in the transposed (sy, sx) orientation.
  3. Same two orientations using the OpenCV legacy ChArUco layout.

If ChArUco refinement fails (e.g. board is partially occluded), the detector
falls back to direct ArUco marker pose estimation using solvePnP over the
known per-marker 3D positions on the board.

This module is intentionally ROS-free: it operates on plain BGR numpy
arrays and a camera intrinsic matrix. The ROS node wraps it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class BoardSpec:
    """ChArUco board geometry.

    ``squares_x`` x ``squares_y`` checker grid, with ArUco markers placed in
    the (col + row) % 2 == 1 white squares (default) or == 0 (legacy).
    """
    squares_x: int = 8
    squares_y: int = 11
    square_length_m: float = 0.029
    marker_length_m: float = 0.021
    dict_id: int = cv2.aruco.DICT_5X5_50


@dataclass
class Detection:
    """Result of a successful board pose estimation."""
    rvec: np.ndarray       # 3x1 Rodrigues rotation vector (board in camera)
    tvec: np.ndarray       # 3x1 translation vector
    method: str            # "charuco" or "aruco"
    variant: str           # human-readable variant tag (e.g. "8x11", "11x8 legacy")
    n_points: int          # number of corners / markers used
    reproj_err_px: float   # mean reprojection error in pixels


def _build_marker_id_to_colrow(sx: int, sy: int, legacy: bool) -> dict:
    """Map ArUco marker ID -> (col, row) for a ChArUco board layout.

    The OpenCV layout assigns marker IDs row-major over the white squares.
    Default layout uses ``(col + row) % 2 == 1`` for marker positions;
    legacy uses ``== 0``.
    """
    mapping = {}
    mid = 0
    for row in range(sy):
        for col in range(sx):
            if legacy:
                is_marker = (col + row) % 2 == 0
            else:
                is_marker = (col + row) % 2 == 1
            if is_marker:
                mapping[mid] = (col, row)
                mid += 1
    return mapping


class BoardDetector:
    """Detect a ChArUco board pose in a single BGR frame.

    Construction is expensive (precompiles four ChArUco/ArUco variants), so
    keep one detector instance for the lifetime of the calibration session.
    """

    def __init__(self, spec: BoardSpec = BoardSpec()):
        self.spec = spec
        self._dictionary = cv2.aruco.getPredefinedDictionary(spec.dict_id)
        self._marker_detector = cv2.aruco.ArucoDetector(self._dictionary)

        # ChArUco variants: 2 orientations x 2 layouts (modern / legacy)
        self._charuco_variants = []
        for sx, sy in [(spec.squares_x, spec.squares_y),
                       (spec.squares_y, spec.squares_x)]:
            for legacy in (False, True):
                board = cv2.aruco.CharucoBoard(
                    (sx, sy), spec.square_length_m,
                    spec.marker_length_m, self._dictionary)
                if legacy:
                    board.setLegacyPattern(True)
                det = cv2.aruco.CharucoDetector(board)
                label = f"{sx}x{sy}" + (" legacy" if legacy else "")
                self._charuco_variants.append((board, det, label))

        # ArUco fallback variants
        self._aruco_variants = []
        for sx, sy in [(spec.squares_x, spec.squares_y),
                       (spec.squares_y, spec.squares_x)]:
            for legacy in (False, True):
                mapping = _build_marker_id_to_colrow(sx, sy, legacy)
                label = f"{sx}x{sy}" + (" legacy" if legacy else "")
                self._aruco_variants.append((mapping, label))

    def detect(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[Detection], np.ndarray]:
        """Detect board pose. Returns (Detection or None, annotated_image)."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        vis = image_bgr.copy()

        marker_corners, marker_ids, _ = self._marker_detector.detectMarkers(gray)
        n_markers = len(marker_ids) if marker_ids is not None else 0
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

        if n_markers < 4:
            cv2.putText(vis, f"ArUco: {n_markers} (need 4+)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            return None, vis

        # ---- Path 1: ChArUco corner refinement ----
        best = self._try_charuco(gray, camera_matrix, dist_coeffs)
        if best is not None:
            det, cc, ci = best
            cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)
            cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                              det.rvec, det.tvec, 0.03)
            cv2.putText(
                vis,
                f"ArUco: {n_markers} | ChArUco: {det.n_points} "
                f"({det.variant}) - READY",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
            return det, vis

        # ---- Path 2: ArUco-marker pose fallback ----
        ar = self._try_aruco_fallback(
            marker_corners, marker_ids, camera_matrix, dist_coeffs)
        if ar is not None:
            cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs,
                              ar.rvec, ar.tvec, 0.03)
            cv2.putText(
                vis,
                f"ArUco fallback: {ar.n_points} markers, "
                f"err={ar.reproj_err_px:.1f}px ({ar.variant}) - READY",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return ar, vis

        cv2.putText(vis, f"ArUco: {n_markers} - no pose",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255), 2)
        return None, vis

    def _try_charuco(self, gray, K, dist):
        """Run ChArUco corner detection across all four variants."""
        best = None
        best_n = 0
        best_label = ""
        best_board = None
        best_cc = None
        best_ci = None
        for board, det, label in self._charuco_variants:
            cc, ci, _, _ = det.detectBoard(gray)
            n = len(cc) if cc is not None else 0
            if n > best_n:
                best_n, best_label, best_board = n, label, board
                best_cc, best_ci = cc, ci

        if best_n < 6 or K is None:
            return None

        obj_pts, img_pts = best_board.matchImagePoints(best_cc, best_ci)
        if obj_pts is None or len(obj_pts) < 6:
            return None

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            return None

        reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(
            img_pts.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1)))
        return Detection(
            rvec=rvec, tvec=tvec, method="charuco",
            variant=best_label, n_points=best_n, reproj_err_px=err,
        ), best_cc, best_ci

    def _try_aruco_fallback(self, marker_corners, marker_ids, K, dist):
        """Direct ArUco marker pose estimation when ChArUco refinement fails."""
        if K is None or marker_ids is None:
            return None
        ids_flat = marker_ids.flatten()
        best = None
        best_err = float("inf")
        spec = self.spec
        for mapping, label in self._aruco_variants:
            obj_pts_list, img_pts_list = [], []
            for i, mid_val in enumerate(ids_flat):
                mid_int = int(mid_val)
                if mid_int not in mapping:
                    continue
                col, row = mapping[mid_int]
                cx = col * spec.square_length_m + spec.square_length_m / 2.0
                cy = row * spec.square_length_m + spec.square_length_m / 2.0
                half = spec.marker_length_m / 2.0
                obj_pts = np.array([
                    [cx - half, cy - half, 0.0],
                    [cx + half, cy - half, 0.0],
                    [cx + half, cy + half, 0.0],
                    [cx - half, cy + half, 0.0],
                ], dtype=np.float64)
                img_pts = marker_corners[i].reshape(4, 2).astype(np.float64)
                obj_pts_list.append(obj_pts)
                img_pts_list.append(img_pts)
            if len(obj_pts_list) < 4:
                continue
            obj_all = np.vstack(obj_pts_list)
            img_all = np.vstack(img_pts_list)
            ok, rvec, tvec = cv2.solvePnP(
                obj_all, img_all, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            reproj, _ = cv2.projectPoints(obj_all, rvec, tvec, K, dist)
            err = float(np.mean(np.linalg.norm(
                img_all - reproj.reshape(-1, 2), axis=1)))
            if err < best_err:
                best_err = err
                best = Detection(
                    rvec=rvec, tvec=tvec, method="aruco",
                    variant=label, n_points=len(obj_pts_list),
                    reproj_err_px=err,
                )
        return best
