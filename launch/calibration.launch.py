"""Launch RealSense + the calibration node.

This is a convenience launch for the common case of a Realsense D435i and
a UR robot that already publishes its TF tree.  If your camera is up via a
separate launch, just run::

    ros2 run ur3e_realsense_handeye calibration_node

instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    info_topic = LaunchConfiguration("camera_info_topic")
    base_frame = LaunchConfiguration("base_frame")
    tool_frame = LaunchConfiguration("tool_frame")
    squares_x = LaunchConfiguration("squares_x")
    squares_y = LaunchConfiguration("squares_y")
    square_length_m = LaunchConfiguration("square_length_m")
    marker_length_m = LaunchConfiguration("marker_length_m")

    return LaunchDescription([
        DeclareLaunchArgument(
            "rgb_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("tool_frame", default_value="tool0"),
        DeclareLaunchArgument("squares_x", default_value="8"),
        DeclareLaunchArgument("squares_y", default_value="11"),
        DeclareLaunchArgument("square_length_m", default_value="0.029"),
        DeclareLaunchArgument("marker_length_m", default_value="0.021"),

        # Bring up the RealSense (color stream + aligned depth, no point cloud).
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="camera",
            namespace="camera",
            output="screen",
            parameters=[{
                "enable_color": True,
                "enable_depth": True,
                "align_depth.enable": True,
                "pointcloud.enable": False,
                "publish_tf": True,
            }],
        ),

        Node(
            package="ur3e_realsense_handeye",
            executable="calibration_node",
            name="handeye_calibration",
            output="screen",
            parameters=[{
                "rgb_topic": rgb_topic,
                "camera_info_topic": info_topic,
                "base_frame": base_frame,
                "tool_frame": tool_frame,
                "squares_x": squares_x,
                "squares_y": squares_y,
                "square_length_m": square_length_m,
                "marker_length_m": marker_length_m,
            }],
        ),
    ])
