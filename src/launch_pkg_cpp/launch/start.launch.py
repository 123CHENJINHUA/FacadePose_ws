#!/usr/bin/env python3
"""
Launch file to start the full system: RealSense camera, IMU odometry, vision pose estimation, and visualization.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare configurable launch arguments
    declare_enable_realsense = DeclareLaunchArgument('enable_realsense', default_value='true', description='Start RealSense camera')

    declare_depth_topic = DeclareLaunchArgument('depth_topic', default_value='camera/camera_middle/aligned_depth_to_color/image_raw')
    declare_color_topic = DeclareLaunchArgument('color_topic', default_value='camera/camera_middle/color/image_raw')
    declare_camera_info_topic = DeclareLaunchArgument('camera_info_topic', default_value='camera/camera_middle/color/camera_info')
    declare_pointcloud_topic = DeclareLaunchArgument('pointcloud_topic', default_value='camera/camera_middle/depth/color/points')
    declare_imu_quat_topic = DeclareLaunchArgument('imu_quat_topic', default_value='/imu/quaternion')

    # New: rosbag play parameters (set here so you can pass them via this launch)
    declare_bag_name = DeclareLaunchArgument('bag_name', default_value='', description='Bag folder name under records; leave empty to use latest')
    declare_rate = DeclareLaunchArgument('rate', default_value='1.0', description='Playback rate for rosbag play')

    # Include RealSense launch (if available)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('launch_pkg_cpp'),
                'launch',
                'realsense_launch.py'
            ])
        ]),
        launch_arguments={
            'enable_realsense': LaunchConfiguration('enable_realsense'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_realsense'))
    )

    # Include Arm Cameras launch
    arm_cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('arm_camera_pkg'),
                'launch',
                'cameras.launch.py'
            ])
        ])
    )

    # IMU odometry node
    imu_node = Node(
        package='imu_pose_pkg',
        executable='imu_pose_node',
        name='imu_pose_node',
        output='screen'
    )

    # Vision pose node
    vision_node = Node(
        package='vision_pose_pkg',
        executable='vision_pose_node',
        name='vision_pose_node',
        parameters=[{
            'depth_topic': LaunchConfiguration('depth_topic'),
            'color_topic': LaunchConfiguration('color_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            'output_topic': '/vision_pose',
            'publish_rate': 30.0,
            'min_plane_points': 300,
            'plane_distance_threshold': 0.02,
            'voxel_size': 0.015,
        }],
        output='screen'
    )

    fusion_node = Node(
        package='fusion_pkg',
        executable='fusion_node',
        name='fusion_node',
        parameters=[{
            'vision_pose_topic': '/vision_pose',
            'imu_quat_topic': LaunchConfiguration('imu_quat_topic'),
            'imu_correct_topic': '/imu_corrected_pose',
            'output_topic': '/fusion_pose',
        }],
        output='screen'
    )

    vis_node = Node(
        package='vis_pkg',
        executable='vis_node',
        name='vision_pose_viz',
        parameters=[{
            'image_topic': LaunchConfiguration('color_topic'),
            'vision_topic': '/vision_pose',
            'imu_correct_topic': '/imu_corrected_pose',
            'fusion_pose_topic': '/fusion_pose',
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
        }],
        output='screen'
    )

    pose_translate_node = Node(
        package='pose_translate_pkg',
        executable='pose_translate_node',
        name='pose_translate_node',
        parameters=[{
            'fusion_pose_topic': '/fusion_pose',
            'output_topic_base1': '/depth_to_pose/base1',
            'output_topic_base2': '/depth_to_pose/base2',
            # 'calibration_yaml': '/path/to/calibration_result.yaml',  # optional override
            'input_frame': 'camera_link',
            'base1_frame': 'base1',
            'base2_frame': 'base2',
        }],
        output='screen'
    )

    comparison_node = Node(
        package='comparison_pkg',
        executable='comparison_node',
        name='pose_comparison_node',
        parameters=[{   
            'vision_topic': '/vision_pose',
            'imu_topic': '/imu_corrected_pose',
            'fusion_topic': '/fusion_pose',
            'save_period': 5.0,
            'output_dir': 'comparison',
            'euler_order': 'xyz',
            'degrees': True,
            'max_points': 10000,
        }],
        output='screen'
    )

    # Include rosbag play with launch arguments; its launch will Shutdown when playback finishes
    def _include_play(context, *args, **kwargs):
        bag_name = LaunchConfiguration('bag_name').perform(context)
        rate = LaunchConfiguration('rate').perform(context)
        launch_args = {}
        if bag_name:
            launch_args['bag_name'] = bag_name
        if rate:
            launch_args['rate'] = rate
        play_include = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('facade_record_pkg'),
                    'launch',
                    'play.launch.py'
                ])
            ]),
            launch_arguments=launch_args.items() if launch_args else None
        )
        return [play_include]

    play_launch = OpaqueFunction(function=_include_play)

    ld = LaunchDescription()

    ld.add_action(declare_enable_realsense)
    ld.add_action(declare_depth_topic)
    ld.add_action(declare_color_topic)
    ld.add_action(declare_camera_info_topic)
    ld.add_action(declare_pointcloud_topic)
    ld.add_action(declare_imu_quat_topic)
    ld.add_action(declare_bag_name)
    ld.add_action(declare_rate)
    ld.add_action(realsense_launch)
    # ld.add_action(arm_cameras_launch)
    ld.add_action(imu_node)
    ld.add_action(vision_node)
    ld.add_action(vis_node)
    ld.add_action(fusion_node)
    # ld.add_action(play_launch)
    # ld.add_action(comparison_node)
    ld.add_action(pose_translate_node)

    return ld


if __name__ == '__main__':
    generate_launch_description()