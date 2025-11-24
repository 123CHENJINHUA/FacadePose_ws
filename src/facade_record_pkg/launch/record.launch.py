from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
import datetime
import os
from pathlib import Path


def generate_launch_description():
    bag_name_default = datetime.datetime.now().strftime("facade_selected_%Y%m%d_%H%M%S")

    file_path = Path(__file__).resolve()
    pkg_name = 'facade_record_pkg'

    # Prefer saving under the SOURCE workspace: <ws>/src/facade_record_pkg/records
    records_dir = None
    for p in file_path.parents:
        if p.name == 'install':
            ws_root = p.parent  # workspace root
            records_dir = ws_root / 'src' / pkg_name / 'records'
            break

    # If not running from install, fall back to package root path
    if records_dir is None:
        # find package dir in the path
        pkg_dir = None
        for p in file_path.parents:
            if p.name == pkg_name:
                pkg_dir = p
                break
        records_dir = (pkg_dir / 'records') if pkg_dir is not None else (file_path.parent / 'records')

    os.makedirs(records_dir, exist_ok=True)
    # Optional: print resolved output directory
    print(f"Recording bags to: {records_dir}")

    topics = [
        '/camera/camera/color/image_raw',
        '/camera/camera/color/camera_info',
        '/camera/camera/aligned_depth_to_color/image_raw',
        '/camera/camera/depth/color/points',
        '/imu/quaternion',
        '/imu/euler_rad',
        '/imu/euler_deg'
    ]

    output_path = PathJoinSubstitution([str(records_dir), LaunchConfiguration('bag_name')])

    return LaunchDescription([
        DeclareLaunchArgument('bag_name', default_value=bag_name_default),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', output_path, *topics],
            output='screen'
        )
    ])
