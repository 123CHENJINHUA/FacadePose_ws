#!/usr/bin/env python3
# launch/realsense_from_yaml.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    enable_realsense = LaunchConfiguration('enable_realsense')

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_realsense',
            default_value='true',
            description='Enable RealSense cameras'
        ),

        # Camera Middle
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ]),
            launch_arguments={
                'camera_name': 'camera_middle',
                'serial_no': "'346222071810'",
                'enable_depth': 'true',
                'enable_color': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'true',
                'rgb_camera.color_profile': '640x480x30',
                'depth_module.depth_profile': '640x480x30',
            }.items(),
            condition=IfCondition(enable_realsense)
        ),

        # Camera Left
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ]),
            launch_arguments={
                'camera_name': 'camera_left',
                'serial_no': "'213522070575'",
                'enable_depth': 'true',
                'enable_color': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'false',
                'rgb_camera.color_profile': '640x480x30',
                'depth_module.depth_profile': '640x480x30',
            }.items(),
            condition=IfCondition(enable_realsense)
        ),

        # Camera Right
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ]),
            launch_arguments={
                'camera_name': 'camera_right',
                'serial_no': "'213622077808'",
                'enable_depth': 'true',
                'enable_color': 'true',
                'align_depth.enable': 'true',
                'pointcloud.enable': 'false',
                'rgb_camera.color_profile': '640x480x30',
                'depth_module.depth_profile': '640x480x30',
            }.items(),
            condition=IfCondition(enable_realsense)
        ),
    ])
