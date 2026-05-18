#!/usr/bin/env python3
"""Top-level ROS 2 node for eye-to-hand calibration.

This is the thin orchestrator that wires together:

  * :mod:`board_detector` (ChArUco/ArUco pose from camera frames)
  * :mod:`solvers`        (hand-eye + non-linear refinement)
  * :mod:`geometry`       (quaternion / RPY conversions for the final output)

Usage:

  1. Attach the ChArUco board to the robot end-effector (tool0).
  2. Bring up the robot driver and the RealSense camera.
  3. ``ros2 run ur3e_realsense_handeye calibration_node``
  4. Put the robot in freedrive and move it to varied poses, keeping the
     board in view of the camera.
  5. Press **c** to capture a sample, **s** to solve, **q** to quit.

Twelve to fifteen well-distributed poses (different rotations matter more
than different positions) typically yields sub-centimeter consistency.
"""

from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
import cv2

from .board_detector import BoardDetector, BoardSpec, Detection
from .solvers import solve_all
from .geometry import (
    rodrigues_to_matrix, transform_msg_to_R_t,
    matrix_to_quaternion, matrix_to_rpy,
)


DEFAULT_BOARD = BoardSpec(
    squares_x=8, squares_y=11,
    square_length_m=0.029, marker_length_m=0.021,
    dict_id=cv2.aruco.DICT_5X5_50,
)


class HandeyeCalibrationNode(Node):
    """Eye-to-hand calibration node.

    Topics (defaults match the RealSense D435i namespaced launch):
      - subs ``rgb_topic``           (sensor_msgs/Image, BGR)
      - subs ``camera_info_topic``   (sensor_msgs/CameraInfo)

    TF lookups: ``base_frame`` -> ``tool_frame`` for each captured sample.
    """

    def __init__(self):
        super().__init__("handeye_calibration")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/camera/camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("squares_x", DEFAULT_BOARD.squares_x)
        self.declare_parameter("squares_y", DEFAULT_BOARD.squares_y)
        self.declare_parameter("square_length_m", DEFAULT_BOARD.square_length_m)
        self.declare_parameter("marker_length_m", DEFAULT_BOARD.marker_length_m)
        self.declare_parameter("min_samples", 5)
        self.declare_parameter("recommend_samples", 12)

        self._rgb_topic = self.get_parameter("rgb_topic").value
        self._info_topic = self.get_parameter("camera_info_topic").value
        self._base_frame = self.get_parameter("base_frame").value
        self._tool_frame = self.get_parameter("tool_frame").value
        self._min_samples = int(self.get_parameter("min_samples").value)
        self._reco_samples = int(self.get_parameter("recommend_samples").value)

        spec = BoardSpec(
            squares_x=int(self.get_parameter("squares_x").value),
            squares_y=int(self.get_parameter("squares_y").value),
            square_length_m=float(self.get_parameter("square_length_m").value),
            marker_length_m=float(self.get_parameter("marker_length_m").value),
            dict_id=DEFAULT_BOARD.dict_id,
        )
        self._detector = BoardDetector(spec)

        self._bridge = CvBridge()
        self._K = None
        self._dist = None
        self._rgb = None
        self._last_detection: Detection | None = None

        # Collected pose pairs
        self._R_target2cam = []
        self._t_target2cam = []
        self._R_gripper2base = []
        self._t_gripper2base = []

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(
            CameraInfo, self._info_topic, self._info_cb, 10)
        self.create_subscription(
            Image, self._rgb_topic, self._rgb_cb, 10)

        self._timer = self.create_timer(1.0 / 30.0, self._display_loop)

        self.get_logger().info(
            "Hand-eye calibration ready.\n"
            f"  Camera: {self._rgb_topic}\n"
            f"  Frames: {self._base_frame} -> {self._tool_frame}\n"
            f"  Board:  {spec.squares_x}x{spec.squares_y}, "
            f"square={spec.square_length_m*1000:.0f}mm, "
            f"marker={spec.marker_length_m*1000:.0f}mm\n"
            "  Keys: 'c'=capture  's'=solve  'q'=quit")

    # ---- Callbacks ----

    def _info_cb(self, msg: CameraInfo):
        self._K = np.array(msg.k).reshape(3, 3)
        self._dist = np.array(msg.d)

    def _rgb_cb(self, msg: Image):
        try:
            self._rgb = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    # ---- Sample management ----

    def _lookup_gripper2base(self):
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._base_frame, self._tool_frame, rclpy.time.Time())
            return transform_msg_to_R_t(tf_msg)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def _capture_sample(self):
        if self._last_detection is None:
            self.get_logger().warn(
                "Board not detected; hold board steady, wait for 'READY', "
                "then press 'c'.")
            return
        robot = self._lookup_gripper2base()
        if robot is None:
            return
        det = self._last_detection
        self._R_target2cam.append(rodrigues_to_matrix(det.rvec))
        self._t_target2cam.append(det.tvec.reshape(3, 1))
        R_g2b, t_g2b = robot
        self._R_gripper2base.append(R_g2b)
        self._t_gripper2base.append(t_g2b)
        n = len(self._R_target2cam)
        self.get_logger().info(
            f"Sample {n} captured ({det.method}, {det.variant}). "
            f"Board t=({det.tvec[0,0]:.3f}, "
            f"{det.tvec[1,0]:.3f}, {det.tvec[2,0]:.3f})")
        if n < self._reco_samples:
            self.get_logger().info(
                f"  At least {self._reco_samples} samples recommended; "
                f"move to a different pose.")

    # ---- Solve & report ----

    def _solve(self):
        n = len(self._R_target2cam)
        if n < self._min_samples:
            self.get_logger().error(
                f"Need at least {self._min_samples} samples (have {n}).")
            return
        self.get_logger().info(f"Solving with {n} samples...")
        results = solve_all(
            self._R_target2cam, self._t_target2cam,
            self._R_gripper2base, self._t_gripper2base,
        )
        if not results:
            self.get_logger().error("All solvers failed.")
            return

        for r in results:
            dist = np.linalg.norm(r.t_cam2base)
            z = r.t_cam2base[2, 0]
            self.get_logger().info(
                f"  {r.name:<11s} consistency={r.consistency_mm:5.1f}mm  "
                f"dist={dist:.3f}m  z={z:+.3f}m")

        best = results[0]
        self._print_result(best)

    def _print_result(self, r):
        cam_x, cam_y, cam_z = r.t_cam2base.flatten()
        cam_dist = float(np.linalg.norm(r.t_cam2base))
        qx, qy, qz, qw = matrix_to_quaternion(r.R_cam2base)
        roll, pitch, yaw = matrix_to_rpy(r.R_cam2base)

        sep = "=" * 62
        log = self.get_logger().info
        log("")
        log(sep)
        log(f"  CALIBRATION RESULT — best method: {r.name}")
        log(sep)
        log(f"  Consistency: {r.consistency_mm:.1f} mm")
        log("  Camera position in base_link frame:")
        log(f"    x = {cam_x:+.4f}  y = {cam_y:+.4f}  z = {cam_z:+.4f}  (m)")
        log(f"  Distance from base: {cam_dist:.3f} m")
        log(f"  Quaternion (base_link -> camera_link):")
        log(f"    x={qx:+.6f}  y={qy:+.6f}  z={qz:+.6f}  w={qw:+.6f}")
        log("")
        log("  Paste into a launch file:")
        log(
            f"  ros2 run tf2_ros static_transform_publisher "
            f"--x {cam_x:.4f} --y {cam_y:.4f} --z {cam_z:.4f} "
            f"--qx {qx:.6f} --qy {qy:.6f} --qz {qz:.6f} --qw {qw:.6f} "
            f"--frame-id {self._base_frame} --child-frame-id camera_link"
        )
        log("")
        log("  Or as URDF joint origin (camera_stand: base -> camera):")
        log(f"    xyz: {cam_x:.4f} {cam_y:.4f} {cam_z:.4f}")
        log(f"    rpy: {roll:.4f} {pitch:.4f} {yaw:.4f}")
        log(sep)

        # Sanity checks
        if cam_dist < 0.1 or cam_dist > 3.0:
            self.get_logger().warn(
                f"Camera distance ({cam_dist:.2f} m) is unusual; "
                f"typical 0.3-1.5 m. Re-check board scale and frames.")
        if cam_z < 0:
            self.get_logger().warn(
                f"Camera Z ({cam_z:.3f} m) is below the robot base; "
                f"verify mounting and base_link orientation.")
        if r.consistency_mm > 20.0:
            self.get_logger().warn(
                f"High consistency error ({r.consistency_mm:.0f} mm). "
                f"Capture more diverse poses (vary rotation, not just position).")

    # ---- Display loop ----

    def _display_loop(self):
        if self._rgb is None or self._K is None:
            return
        det, vis = self._detector.detect(self._rgb, self._K, self._dist)
        self._last_detection = det

        n = len(self._R_target2cam)
        cv2.putText(
            vis,
            f"Samples: {n}/{self._reco_samples}+    "
            f"keys: 'c'=capture  's'=solve  'q'=quit",
            (20, vis.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Hand-Eye Calibration (eye-to-hand)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self._capture_sample()
        elif key == ord('s'):
            self._solve()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            self.get_logger().info("Quitting.")
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = HandeyeCalibrationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
