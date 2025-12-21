from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='arm_camera_pkg',
            executable='camera_node',
            name='rgb_camera1',
            parameters=[{
                'device_id': 6,
                'topic_name': 'rgb_camera1/image_raw',
                'frame_id': 'rgb_camera1_link',
                'width': 640,
                'height': 480,
                'frame_rate': 30
            }]
        ),
        Node(
            package='arm_camera_pkg',
            executable='camera_node',
            name='rgb_camera2',
            parameters=[{
                'device_id': 8,  
                'topic_name': 'rgb_camera2/image_raw',
                'frame_id': 'rgb_camera2_link',
                'width': 640,
                'height': 480,
                'frame_rate': 30
            }]
        )
    ])
