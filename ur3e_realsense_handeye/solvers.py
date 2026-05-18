"""Hand-eye calibration solvers (eye-to-hand setup).

The camera is fixed in the world; the calibration board is rigidly attached
to the robot end-effector (tool0). For each captured sample we know:

  * T_target2cam : pose of the board in the camera frame (from PnP)
  * T_gripper2base : pose of the EE in the robot base frame (from TF)

The unknowns we solve for are:

  * T_cam2base : the constant pose of the camera in the robot base frame
  * T_target2gripper : the constant pose of the board on the EE

In OpenCV's eye-in-hand formulation ``calibrateHandEye`` finds X such that
``A_i * X = X * B_i`` from synchronized motion of A (target-in-camera) and
B (gripper-in-base). For eye-to-hand the inputs are swapped so the recovered
X equals T_gripper2target, and the camera-in-base transform is reconstructed
per-sample and averaged for robustness.

Solvers wrapped here:

  * cv2.calibrateHandEye in TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS modes
  * cv2.calibrateRobotWorldHandEye in SHAH and LI modes
  * A 12-parameter non-linear least-squares refinement using scipy that
    jointly optimizes T_base2cam and T_target2gripper against the observed
    target-in-camera measurements.

Each solver returns a :class:`SolverResult`. The :func:`solve_all` helper
runs every solver, scores them by per-sample camera-position consistency
(standard deviation across the recovered T_cam2base translations), and
returns a sorted list.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from scipy.optimize import least_squares as scipy_least_squares

from .geometry import compose_transforms, invert_transform


@dataclass
class SolverResult:
    name: str
    R_cam2base: np.ndarray  # 3x3
    t_cam2base: np.ndarray  # 3x1
    consistency_mm: float   # mean per-axis stddev of camera position across samples


def _compute_cam2base_from_X(
    R_X, t_X,
    R_target2cam_list, t_target2cam_list,
    R_gripper2base_list, t_gripper2base_list,
) -> Optional[SolverResult]:
    """Given X = T_gripper2target, derive T_cam2base from every sample.

    Each sample produces an estimate; we use the per-sample standard
    deviation as a consistency score, and pick the rotation from the
    sample whose camera-position is closest to the mean.
    """
    n = len(R_target2cam_list)
    cam_positions = []

    for i in range(n):
        R_t2c, t_t2c = R_target2cam_list[i], t_target2cam_list[i]
        R_g2b, t_g2b = R_gripper2base_list[i], t_gripper2base_list[i]
        R_b2g, t_b2g = invert_transform(R_g2b, t_g2b)

        # T_base2cam = T_target2cam * X * T_base2gripper
        R_xb, t_xb = compose_transforms(R_X, t_X, R_b2g, t_b2g)
        R_b2c, t_b2c = compose_transforms(R_t2c, t_t2c, R_xb, t_xb)
        R_c2b, t_c2b = invert_transform(R_b2c, t_b2c)
        cam_positions.append(t_c2b.flatten())

    pos_arr = np.array(cam_positions)
    consistency = float(np.mean(np.std(pos_arr, axis=0)) * 1000.0)
    t_avg = np.mean(pos_arr, axis=0).reshape(3, 1)

    # Use the rotation from the sample whose camera position is closest to mean
    dists = np.linalg.norm(pos_arr - t_avg.flatten(), axis=1)
    median_idx = int(np.argmin(dists))
    R_g2b_med = R_gripper2base_list[median_idx]
    t_g2b_med = t_gripper2base_list[median_idx]
    R_b2g_med, t_b2g_med = invert_transform(R_g2b_med, t_g2b_med)
    R_xb, t_xb = compose_transforms(R_X, t_X, R_b2g_med, t_b2g_med)
    R_b2c, t_b2c = compose_transforms(
        R_target2cam_list[median_idx],
        t_target2cam_list[median_idx],
        R_xb, t_xb,
    )
    R_c2b, _ = invert_transform(R_b2c, t_b2c)
    return SolverResult("", R_c2b, t_avg, consistency)


def solve_calibrate_handeye(
    R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list,
) -> List[SolverResult]:
    """Run cv2.calibrateHandEye in all five OpenCV modes (eye-to-hand swap)."""
    out = []
    methods = [
        ("Tsai",      cv2.CALIB_HAND_EYE_TSAI),
        ("Park",      cv2.CALIB_HAND_EYE_PARK),
        ("Horaud",    cv2.CALIB_HAND_EYE_HORAUD),
        ("Andreff",   cv2.CALIB_HAND_EYE_ANDREFF),
        ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]
    for name, method in methods:
        try:
            R_X, t_X = cv2.calibrateHandEye(
                # eye-to-hand swap
                R_t2c_list, t_t2c_list,
                R_g2b_list, t_g2b_list,
                method=method,
            )
        except cv2.error:
            continue
        res = _compute_cam2base_from_X(
            R_X, t_X, R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list)
        if res is not None:
            res.name = name
            out.append(res)
    return out


def solve_robot_world_handeye(
    R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list,
) -> List[SolverResult]:
    """cv2.calibrateRobotWorldHandEye in SHAH and LI modes."""
    R_b2g_list = [R.T for R in R_g2b_list]
    t_b2g_list = [(-R.T @ t) for R, t in zip(R_g2b_list, t_g2b_list)]
    methods = [
        ("RW-Shah", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
        ("RW-Li",   cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI),
    ]
    out = []
    n = len(R_t2c_list)
    for name, method in methods:
        try:
            R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
                R_t2c_list, t_t2c_list,
                R_b2g_list, t_b2g_list,
                method=method,
            )
        except cv2.error:
            continue

        # In eye-to-hand: Z = T_base2cam = (R_g2c, t_g2c) -> T_cam2base = inv(Z)
        R_c2b = R_g2c.T
        t_c2b = -R_g2c.T @ t_g2c

        cam_positions = []
        for i in range(n):
            R_xb, t_xb = compose_transforms(
                R_b2w, t_b2w, R_g2b_list[i], t_g2b_list[i])
            R_b2c, t_b2c = compose_transforms(
                R_t2c_list[i], t_t2c_list[i], R_xb, t_xb)
            R_ci, t_ci = invert_transform(R_b2c, t_b2c)
            cam_positions.append(t_ci.flatten())
        pos_arr = np.array(cam_positions)
        consistency = float(np.mean(np.std(pos_arr, axis=0)) * 1000.0)
        out.append(SolverResult(name, R_c2b, t_c2b, consistency))
    return out


def solve_nonlinear(
    R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list,
    init_R_c2b: np.ndarray, init_t_c2b: np.ndarray,
) -> Optional[Tuple[SolverResult, float]]:
    """Jointly optimize T_base2cam and T_target2gripper via Levenberg-Marquardt.

    The 12-DOF parameter vector is::

      [rvec_b2c (3), tvec_b2c (3), rvec_t2g (3), tvec_t2g (3)]

    For each sample we form the predicted ``T_target2cam`` from the current
    estimates and stack the rotation-vector and translation residuals against
    the observed value. Returns ``(SolverResult, rms_translation_mm)`` on
    success, else ``None``.
    """
    n = len(R_t2c_list)
    if n < 3:
        return None

    init_R_b2c, init_t_b2c = invert_transform(init_R_c2b, init_t_c2b)

    # Derive initial T_target2gripper from the first sample.
    R_g2b_0, t_g2b_0 = R_g2b_list[0], t_g2b_list[0]
    R_b2g_0, t_b2g_0 = invert_transform(R_g2b_0, t_g2b_0)
    R_cb_tc, t_cb_tc = compose_transforms(
        init_R_c2b, init_t_c2b, R_t2c_list[0], t_t2c_list[0])
    init_R_t2g, init_t_t2g = compose_transforms(
        R_b2g_0, t_b2g_0, R_cb_tc, t_cb_tc)

    rvec_b2c_init, _ = cv2.Rodrigues(init_R_b2c)
    rvec_t2g_init, _ = cv2.Rodrigues(init_R_t2g)
    x0 = np.concatenate([
        rvec_b2c_init.flatten(),
        init_t_b2c.flatten(),
        rvec_t2g_init.flatten(),
        init_t_t2g.flatten(),
    ])

    def residuals(x):
        rv_b2c = x[0:3].reshape(3, 1)
        tv_b2c = x[3:6].reshape(3, 1)
        rv_t2g = x[6:9].reshape(3, 1)
        tv_t2g = x[9:12].reshape(3, 1)
        R_b2c, _ = cv2.Rodrigues(rv_b2c)
        R_t2g, _ = cv2.Rodrigues(rv_t2g)

        res = []
        for i in range(n):
            R_tmp, t_tmp = compose_transforms(
                R_b2c, tv_b2c, R_g2b_list[i], t_g2b_list[i])
            R_pred, t_pred = compose_transforms(R_tmp, t_tmp, R_t2g, tv_t2g)

            R_obs, t_obs = R_t2c_list[i], t_t2c_list[i]
            R_err = R_pred.T @ R_obs
            rv_err, _ = cv2.Rodrigues(R_err)
            t_err = t_pred - t_obs
            res.extend(rv_err.flatten().tolist())
            res.extend(t_err.flatten().tolist())
        return np.array(res)

    try:
        opt = scipy_least_squares(
            residuals, x0, method="lm",
            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000)
    except Exception:
        return None

    xopt = opt.x
    rv_b2c = xopt[0:3].reshape(3, 1)
    tv_b2c = xopt[3:6].reshape(3, 1)
    R_b2c_opt, _ = cv2.Rodrigues(rv_b2c)
    R_c2b_opt = R_b2c_opt.T
    t_c2b_opt = -R_b2c_opt.T @ tv_b2c

    rv_t2g = xopt[6:9].reshape(3, 1)
    tv_t2g = xopt[9:12].reshape(3, 1)
    R_t2g_opt, _ = cv2.Rodrigues(rv_t2g)

    cam_positions = []
    for i in range(n):
        R_b2g_i, t_b2g_i = invert_transform(R_g2b_list[i], t_g2b_list[i])
        R_g2t, t_g2t = invert_transform(R_t2g_opt, tv_t2g)
        R_tmp, t_tmp = compose_transforms(
            R_t2c_list[i], t_t2c_list[i], R_g2t, t_g2t)
        R_b2c_i, t_b2c_i = compose_transforms(R_tmp, t_tmp, R_b2g_i, t_b2g_i)
        R_c2b_i, t_c2b_i = invert_transform(R_b2c_i, t_b2c_i)
        cam_positions.append(t_c2b_i.flatten())
    pos_arr = np.array(cam_positions)
    consistency = float(np.mean(np.std(pos_arr, axis=0)) * 1000.0)

    final_res = residuals(xopt)
    t_residuals = [np.linalg.norm(final_res[i * 6 + 3 : i * 6 + 6])
                   for i in range(n)]
    rms_t_mm = float(np.sqrt(np.mean(np.array(t_residuals) ** 2)) * 1000.0)

    return SolverResult("NL-Optim", R_c2b_opt, t_c2b_opt, consistency), rms_t_mm


def solve_all(
    R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list,
) -> List[SolverResult]:
    """Run every solver and return results sorted by consistency_mm ascending.

    The non-linear refinement is seeded with the best linear-solver result,
    so it always runs last and usually wins.
    """
    results = []
    results.extend(solve_calibrate_handeye(
        R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list))
    results.extend(solve_robot_world_handeye(
        R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list))
    if not results:
        return []

    # Seed nonlinear with the best linear result.
    results.sort(key=lambda r: r.consistency_mm)
    seed = results[0]
    nl = solve_nonlinear(
        R_t2c_list, t_t2c_list, R_g2b_list, t_g2b_list,
        seed.R_cam2base, seed.t_cam2base,
    )
    if nl is not None:
        results.append(nl[0])

    results.sort(key=lambda r: r.consistency_mm)
    return results
