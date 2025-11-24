from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.event_handlers import OnProcessExit
import datetime
import os
from pathlib import Path


def _resolve_records_dir(current_file: Path, pkg_name: str) -> Path:
    # Prefer saving under the SOURCE workspace: <ws>/src/<pkg_name>/records
    for p in current_file.parents:
        if p.name == 'install':
            ws_root = p.parent  # workspace root
            return ws_root / 'src' / pkg_name / 'records'
    # Otherwise (running from source), place next to package root
    for p in current_file.parents:
        if p.name == pkg_name:
            return p / 'records'
    return current_file.parent / 'records'


def _latest_bag_name(records_dir: Path) -> str:
    try:
        if not records_dir.exists():
            return datetime.datetime.now().strftime("facade_play_%Y%m%d_%H%M%S")
        candidates = [d for d in records_dir.iterdir() if d.is_dir()]
        if not candidates:
            return datetime.datetime.now().strftime("facade_play_%Y%m%d_%H%M%S")
        latest = max(candidates, key=lambda d: d.stat().st_mtime)
        return latest.name
    except Exception:
        return datetime.datetime.now().strftime("facade_play_%Y%m%d_%H%M%S")


def generate_launch_description():
    pkg_name = 'facade_record_pkg'
    file_path = Path(__file__).resolve()
    records_dir = _resolve_records_dir(file_path, pkg_name)
    os.makedirs(records_dir, exist_ok=True)

    default_bag_name = _latest_bag_name(records_dir)

    # Info output
    print(f"Playing bags from: {records_dir}")
    print(f"Default bag folder: {default_bag_name}")

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '-r', LaunchConfiguration('rate'),
             PathJoinSubstitution([str(records_dir), LaunchConfiguration('bag_name')])],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('bag_name', default_value=default_bag_name,
                              description='Bag folder name under records to play'),
        DeclareLaunchArgument('rate', default_value='1.0', description='Playback rate'),
        bag_play,
        RegisterEventHandler(
            OnProcessExit(target_action=bag_play, on_exit=[Shutdown(reason='rosbag play finished')])
        )
    ])
