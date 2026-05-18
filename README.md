# UR3e + RealSense Eye-to-Hand Calibration

ROS 2 Humble package that recovers the rigid transform from a **fixed Intel
RealSense D435i** to a **Universal Robots UR3e** base frame, using a ChArUco
board attached to the robot end-effector.

It runs **eight solvers** on the same captured sample set (five OpenCV
hand-eye methods, two robot-world hand-eye methods, and a 12-DOF
non-linear refinement) and picks the one with the lowest per-sample
camera-position consistency. A typical 12-sample calibration converges to
**< 10 mm consistency** with a varied set of board poses.

> The package is hardware-agnostic: any UR-style arm publishing
> `base_link → tool0` and any color camera publishing `image_raw +
> camera_info` will work. UR3e and D435i are the defaults because that's
> the rig the package was developed and validated on.

## Why this exists

A common eye-to-hand workflow ends with `cv2.calibrateHandEye()` and a
single transform. That works, but a few pitfalls bite in practice:

- A single solver can return a degenerate result on noisy data and you
  won't notice without an independent check.
- The OpenCV API for eye-in-hand vs. eye-to-hand is the same call with
  inputs swapped, which is easy to get wrong silently.
- ChArUco detection drops out the moment a marker is occluded; without a
  fallback you lose entire samples.
- The final TF is needed in two forms (quaternion for
  `static_transform_publisher`, RPY for URDF), and pasting numbers by hand
  is error-prone.

This package addresses all four:

| Concern                              | How it's handled                                                                                            |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Trusting a single solver             | Runs eight, scores by per-sample camera-position stddev, picks the winner.                                  |
| Eye-to-hand vs eye-in-hand confusion | The eye-to-hand input swap is encoded in `solvers.py` and documented in the module docstring.               |
| ChArUco drops markers                | Tries four ChArUco variants (orientation × legacy/modern), then falls back to direct ArUco PnP if all fail. |
| Hand-pasting the transform           | Prints both the `ros2 run tf2_ros static_transform_publisher` command and the URDF `xyz`/`rpy` block.       |

## Package layout

```
ur3e_realsense_handeye/
├── geometry.py          # rigid-transform utilities (pure math, no ROS)
├── board_detector.py    # ChArUco + ArUco-fallback pose estimation
├── solvers.py           # 5 HandEye + 2 RobotWorldHandEye + NL refinement
├── calibration_node.py  # ROS 2 orchestrator (subscribes camera, looks up TF, runs solvers)
└── board_generator.py   # standalone script: print a ChArUco PNG at the right scale
launch/
└── calibration.launch.py  # RealSense bring-up + calibration node
package.xml / setup.py
```

The split keeps the math testable (no ROS imports in `geometry`,
`board_detector`, or `solvers`) and lets you reuse the solvers in a
non-ROS workflow if needed.

## Setup (Ubuntu 22.04 + ROS 2 Humble)

```bash
sudo apt update
sudo apt install -y \
  ros-humble-realsense2-camera \
  ros-humble-realsense2-description \
  ros-humble-tf2-ros \
  ros-humble-cv-bridge \
  ros-humble-rviz2 \
  python3-colcon-common-extensions python3-rosdep
pip3 install "numpy<2" "opencv-python<4.11" scipy

# Build
mkdir -p ~/calib_ws/src && cd ~/calib_ws/src
git clone https://github.com/NishanthSundaran/UR3E-Camera-calibration-handeye-Intel-realsense.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

> `numpy` must be **<2** and `opencv-python` **<4.11**. ROS 2 Humble's
> `cv_bridge` segfaults under newer numpy/OpenCV.

## Usage

### 1. Generate and print the board

```bash
ros2 run ur3e_realsense_handeye board_generator \
    --squares_x 8 --squares_y 11 \
    --square_length_mm 29 --marker_length_mm 21 \
    --dpi 300 --out charuco_8x11.png
```

Print at **100% scale** (no fit-to-page), then **measure the actual
printed squares with a ruler** before mounting. Printer scaling errors
go directly into the calibration.

Tape or glue the board to a flat plate, then attach the plate rigidly to
the gripper or end-effector (`tool0`).

### 2. Bring up robot + camera

The robot driver must publish `base_link → tool0` in TF, and the camera
must publish `image_raw + camera_info`. For a UR3e + RealSense rig the
default topics already match.

### 3. Run the calibration

If your camera isn't running yet:

```bash
ros2 launch ur3e_realsense_handeye calibration.launch.py
```

If the camera is already up:

```bash
ros2 run ur3e_realsense_handeye calibration_node
```

A live preview window opens with detected markers and the estimated board
axes. Once you see green **READY** text:

1. Put the robot in **freedrive** (teach pendant).
2. Move it to a pose where the board is fully visible and well-lit.
3. Press **`c`** to capture a sample.
4. Move to a *different* orientation (rotation matters more than
   translation), repeat.
5. After **≥ 12 samples** with varied rotations, press **`s`** to solve.
6. Press **`q`** to quit.

Override the board geometry or topics at the command line:

```bash
ros2 run ur3e_realsense_handeye calibration_node --ros-args \
    -p rgb_topic:=/my_camera/image_raw \
    -p camera_info_topic:=/my_camera/camera_info \
    -p base_frame:=base_link \
    -p tool_frame:=tool0 \
    -p squares_x:=8 -p squares_y:=11 \
    -p square_length_m:=0.029 -p marker_length_m:=0.021
```

### 4. Read the result

After pressing **`s`**, the node prints a result block like:

```
======================================================================
  CALIBRATION RESULT (best method: NL-Optim)
======================================================================
  Consistency: 4.2 mm
  Camera position in base_link frame:
    x = +0.0137  y = +1.0702  z = +0.9151  (m)
  Distance from base: 1.420 m
  Quaternion (base_link -> camera_link):
    x=+0.207982  y=+0.210572  z=-0.675404  w=+0.675449

  Paste into a launch file:
  ros2 run tf2_ros static_transform_publisher \
      --x 0.0137 --y 1.0702 --z 0.9151 \
      --qx 0.207982 --qy 0.210572 --qz -0.675404 --qw 0.675449 \
      --frame-id base_link --child-frame-id camera_link

  Or as URDF joint origin (camera_stand: base -> camera):
    xyz: 0.0137 1.0702 0.9151
    rpy: -0.0123 0.4521 -1.5707
======================================================================
```

The node also evaluates and prints every solver's score so you can see
how they compare on your data.

## Tips for a good calibration

- **Vary rotation, not just translation.** Sliding the EE in a plane
  produces near-degenerate samples for `calibrateHandEye`. Tilt the EE
  ±30° about each axis between captures.
- **Cover the camera frame.** Move the board into corners and centre,
  near and far, so the solvers see the full intrinsic effects.
- **Hold steady before pressing `c`.** Motion blur during PnP biases the
  per-sample board pose.
- **Watch the consistency metric.** Under 10 mm is typical, under 5 mm
  with non-linear refinement on a printed board, < 2 mm with a CNC plate.
  If you're stuck > 20 mm, capture more rotation diversity.

## How the solvers compare

| Solver        | Source                              | Notes                                                                 |
|---------------|-------------------------------------|-----------------------------------------------------------------------|
| `Tsai`        | OpenCV `CALIB_HAND_EYE_TSAI`        | Closed-form, fast, sensitive to noise.                                |
| `Park`        | OpenCV `CALIB_HAND_EYE_PARK`        | Closed-form, decouples rotation and translation.                       |
| `Horaud`      | OpenCV `CALIB_HAND_EYE_HORAUD`      | Closed-form, dual-quaternion-like.                                     |
| `Andreff`     | OpenCV `CALIB_HAND_EYE_ANDREFF`     | Linear, treats rotation and translation jointly.                       |
| `Daniilidis`  | OpenCV `CALIB_HAND_EYE_DANIILIDIS`  | Dual quaternion, often the best linear solver.                         |
| `RW-Shah`     | `CALIB_ROBOT_WORLD_HAND_EYE_SHAH`   | Robot-world formulation; jointly recovers `T_base2world` and `T_g2c`.  |
| `RW-Li`       | `CALIB_ROBOT_WORLD_HAND_EYE_LI`     | Same but with Li's closed-form.                                        |
| `NL-Optim`    | This package (scipy LM)             | 12-DOF refinement, seeded with the best linear result; usually wins.    |

The "winner" is whichever solver produces the smallest per-sample
camera-position stddev across all captured samples, i.e. the most
*internally consistent* answer, not necessarily the one with the lowest
single-sample reprojection error.

## License

Apache 2.0. See LICENSE.

## Author

**Nishanth Sundaran** ([sundharnishanth@gmail.com](mailto:sundharnishanth@gmail.com))

Originally written as part of an M.Eng thesis on force-feedback HRI on a
UR3e; extracted into this standalone package for reuse.
