#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose translation node:
- Subscribes: /fusion_pose (PoseStamped) in camera/depth frame
- Auto-discovers calibration_result.yaml (embedded 4x4 transforms) inside this package
- Publishes: /depth_to_pose/base1 and /depth_to_pose/base2 (PoseStamped)
"""

import os
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class PoseTranslateNode(Node):
    def __init__(self):
        super().__init__('pose_translate_node')
        # Parameters
        self.declare_parameter('fusion_pose_topic', '/fusion_pose')
        self.declare_parameter('output_topic_base1', '/depth_to_pose/base1')
        self.declare_parameter('output_topic_base2', '/depth_to_pose/base2')
        # Allow overriding, but default empty to trigger auto-discovery
        self.declare_parameter('calibration_yaml', '')
        self.declare_parameter('input_frame', 'camera_link')
        self.declare_parameter('base1_frame', 'base1')
        self.declare_parameter('base2_frame', 'base2')

        self.fusion_pose_topic = self.get_parameter('fusion_pose_topic').value
        self.output_topic_base1 = self.get_parameter('output_topic_base1').value
        self.output_topic_base2 = self.get_parameter('output_topic_base2').value
        user_yaml_param = self.get_parameter('calibration_yaml').value
        self.input_frame = self.get_parameter('input_frame').value
        self.base1_frame = self.get_parameter('base1_frame').value
        self.base2_frame = self.get_parameter('base2_frame').value

        # Auto-locate YAML if parameter not provided
        self.calibration_yaml = self._discover_yaml(user_yaml_param)
        self.get_logger().info(f'Calibration YAML resolved to: {self.calibration_yaml}')

        # Load transforms
        self.T_cam_base1, self.T_cam_base2 = self._load_transforms(self.calibration_yaml)
        self.get_logger().info('Loaded camera->base transforms (embedded)')
        self.get_logger().info(f" T_cam_base1:\n{self.T_cam_base1}")
        self.get_logger().info(f" T_cam_base2:\n{self.T_cam_base2}")

        # Publishers
        self.pub_base1 = self.create_publisher(PoseStamped, self.output_topic_base1, 10)
        self.pub_base2 = self.create_publisher(PoseStamped, self.output_topic_base2, 10)

        # Additional publishers for Position (x,y,z) and Euler angles
        self.pub_base1_pos = self.create_publisher(Vector3Stamped, self.output_topic_base1 + '/position', 10)
        self.pub_base1_euler = self.create_publisher(Vector3Stamped, self.output_topic_base1 + '/euler_deg', 10)
        
        self.pub_base2_pos = self.create_publisher(Vector3Stamped, self.output_topic_base2 + '/position', 10)
        self.pub_base2_euler = self.create_publisher(Vector3Stamped, self.output_topic_base2 + '/euler_deg', 10)

        # Subscriber
        self.create_subscription(PoseStamped, self.fusion_pose_topic, self.fusion_cb, 50)
        self.get_logger().info(f"Subscribed to fusion pose: {self.fusion_pose_topic}")
        self.get_logger().info(f"Publishing to {self.output_topic_base1} and {self.output_topic_base2}")

    def _discover_yaml(self, override: str) -> str:
        if override and os.path.exists(override):
            return override
        file_path = Path(__file__).resolve()
        # Try development layout: <ws>/src/pose_translate_pkg/calibration_result/calibration_result.yaml
        for p in file_path.parents:
            if p.name == 'src':
                candidate = p / 'pose_translate_pkg' / 'calibration_result' / 'calibration_result.yaml'
                if candidate.exists():
                    return str(candidate)
        # Try installed layout: <ws>/install/pose_translate_pkg/share/pose_translate_pkg/calibration_result/calibration_result.yaml
        for p in file_path.parents:
            if p.name == 'install':
                ws_root = p.parent
                candidate = ws_root / 'src' / 'pose_translate_pkg' / 'calibration_result' / 'calibration_result.yaml'
                if candidate.exists():
                    return str(candidate)
        raise FileNotFoundError('Unable to auto-locate calibration_result.yaml; provide calibration_yaml parameter explicitly.')

    def _load_transforms(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Calibration YAML not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        m1 = data.get('base1_matrix')
        m2 = data.get('base2_matrix')
        if m1 is None or m2 is None:
            raise ValueError('Missing base1_matrix or base2_matrix in YAML')
        T1 = np.array(m1, dtype=np.float64)
        T2 = np.array(m2, dtype=np.float64)
        if T1.shape != (4,4) or T2.shape != (4,4):
            raise ValueError('Matrices must be 4x4')
        return T1, T2

    def pose_to_T(self,msg: PoseStamped):
        q = msg.pose.orientation
        t = msg.pose.position
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        Rm = R.from_quat(quat).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = Rm
        T[0:3, 3] = np.array([t.x*1000, t.y*1000, t.z*1000], dtype=np.float64)
        return T

    def T_to_pose(self,T: np.ndarray, frame_id: str, stamp):
        Rm = T[0:3, 0:3]
        t = T[0:3, 3]
        quat = R.from_matrix(Rm).as_quat()  # [x,y,z,w]
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        return msg

    def fusion_cb(self, msg: PoseStamped):
        try:
            T_cam = self.pose_to_T(msg)
            stamp = msg.header.stamp

            # Apply camera->base transforms: T_base = T_cam_base * T_cam
            T_b1 = self.T_cam_base1.T @ T_cam
            T_b2 = self.T_cam_base2.T @ T_cam

            msg_b1 = self.T_to_pose(T_b1, self.base1_frame, stamp)
            msg_b2 = self.T_to_pose(T_b2, self.base2_frame, stamp)

            self.pub_base1.publish(msg_b1)
            self.pub_base2.publish(msg_b2)

            # Publish extra topics
            self.publish_extra(T_b1, self.base1_frame, stamp, self.pub_base1_pos, self.pub_base1_euler)
            self.publish_extra(T_b2, self.base2_frame, stamp, self.pub_base2_pos, self.pub_base2_euler)

        except Exception as e:
            self.get_logger().warn(f"Pose translation failed: {e}")

    def publish_extra(self, T, frame_id, stamp, pub_pos, pub_euler):
        Rm = T[0:3, 0:3]
        t = T[0:3, 3]
        # Euler in degrees
        euler = R.from_matrix(Rm).as_euler('xyz', degrees=True)
        
        # Position
        pos_msg = Vector3Stamped()
        pos_msg.header.stamp = stamp
        pos_msg.header.frame_id = frame_id
        pos_msg.vector.x = float(t[0])
        pos_msg.vector.y = float(t[1])
        pos_msg.vector.z = float(t[2])
        pub_pos.publish(pos_msg)
        
        # Euler
        euler_msg = Vector3Stamped()
        euler_msg.header.stamp = stamp
        euler_msg.header.frame_id = frame_id
        euler_msg.vector.x = float(euler[0])
        euler_msg.vector.y = float(euler[1])
        euler_msg.vector.z = float(euler[2])
        pub_euler.publish(euler_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseTranslateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
