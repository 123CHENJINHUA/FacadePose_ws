#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合节点：将 IMU 与 Vision 的姿态进行互补滤波 (四元数球面插值)，发布 /fusion_pose
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from std_msgs.msg import Header
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
from scipy.spatial.transform import Rotation as R


class FusionPoseNode(Node):
    def __init__(self):
        super().__init__('fusion_pose_node')

        # 参数
        self.declare_parameter('vision_pose_topic', '/vision_pose')
        self.declare_parameter('imu_quat_topic', 'imu/quaternion')
        self.declare_parameter('output_topic', '/fusion_pose')
        self.declare_parameter('imu_correct_topic', '/imu_corrected_pose')
        self.declare_parameter('output_frame', 'camera_link')  # 与视觉保持一致或根据需求修改
        self.declare_parameter('imu_weight', 0.7)              # IMU 权重 (高频) 0~1
        self.declare_parameter('smooth_factor', 0.3)           # 对融合结果再平滑 (0 不平滑)
        self.declare_parameter('sync_queue', 30)
        self.declare_parameter('sync_slop', 0.05)

        self.vision_pose_topic = self.get_parameter('vision_pose_topic').value
        self.imu_quat_topic = self.get_parameter('imu_quat_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.imu_correct_topic = self.get_parameter('imu_correct_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.imu_weight = float(self.get_parameter('imu_weight').value)
        self.smooth_factor = float(self.get_parameter('smooth_factor').value)
        self.sync_queue = int(self.get_parameter('sync_queue').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        # 与 vis_node 一致的 IMU-相机外参
        self.setup_imu_camera_transform()

        # 存储初始姿态
        self.camera_initial_rmat = None   # 第一帧视觉姿态的旋转矩阵
        self.imu_initial_rmat = None      # 第一帧 IMU 四元数对应的旋转矩阵

        # 订阅 & 同步
        self.vis_sub = Subscriber(self, PoseStamped, self.vision_pose_topic)
        self.imu_sub = Subscriber(self, QuaternionStamped, self.imu_quat_topic)
        self.ts = ApproximateTimeSynchronizer(
            [self.vis_sub, self.imu_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=False
        )
        self.ts.registerCallback(self.sync_callback)

        # 发布者
        self.pose_pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.imu_correct_pub = self.create_publisher(PoseStamped, self.imu_correct_topic, 10)

        # 状态
        self.last_fused_quat = None  # 保存上一次融合四元数 [x,y,z,w]
        self.frame_count = 0
        self.get_logger().info('FusionPoseNode started')
        self.get_logger().info(f' vision_pose_topic: {self.vision_pose_topic}')
        self.get_logger().info(f' imu_quat_topic: {self.imu_quat_topic}')
        self.get_logger().info(f' output_topic: {self.output_topic}')
        self.get_logger().info(f' imu_weight: {self.imu_weight:.2f}, smooth_factor: {self.smooth_factor:.2f}')

    def setup_imu_camera_transform(self):
        # camera -> imu
        self.R_ic = np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=np.float64)
        # imu -> camera (transpose/inverse)
        self.R_ci = self.R_ic.T

    # 球面插值 (Slerp) weight ∈ [0,1]; 返回归一化四元数
    def slerp(self, q0, q1, weight):
        q0 = np.array(q0, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)
        # 确保同向，避免反转导致最短路径错误
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        # 如果角度很小，用线性插值近似
        if dot > 0.9995:
            q = q0 + weight * (q1 - q0)
            return (q / np.linalg.norm(q)).tolist()
        theta_0 = np.arccos(dot)
        theta = theta_0 * weight
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        q = (s0 * q0) + (s1 * q1)
        return (q / np.linalg.norm(q)).tolist()

    def sync_callback(self, vision_pose: PoseStamped, imu_quat: QuaternionStamped):
        try:
            # 提取并归一化四元数 (x,y,z,w)
            q_vis = [
                vision_pose.pose.orientation.x,
                vision_pose.pose.orientation.y,
                vision_pose.pose.orientation.z,
                vision_pose.pose.orientation.w,
            ]
            q_imu = [
                imu_quat.quaternion.x,
                imu_quat.quaternion.y,
                imu_quat.quaternion.z,
                imu_quat.quaternion.w,
            ]
            def norm_q(q):
                q = np.array(q, dtype=np.float64)
                n = np.linalg.norm(q)
                return (q / n).tolist() if n > 1e-12 else [0., 0., 0., 1.]
            q_vis = norm_q(q_vis)
            q_imu = norm_q(q_imu)

            # 转旋转矩阵
            R_vis = R.from_quat(q_vis).as_matrix()
            R_imu = R.from_quat(q_imu).as_matrix()

            # 记录初始姿态（与 vis_node 一致）
            if self.camera_initial_rmat is None:
                self.camera_initial_rmat = R_vis.copy()
            if self.imu_initial_rmat is None:
                self.imu_initial_rmat = R_imu.copy()

            # 将 IMU 的相对旋转映射到相机坐标系，并作用到相机初始姿态，得到与视觉同坐标系、同参考的 IMU 姿态
            R_delta_imu = self.imu_initial_rmat.T @ R_imu           # IMU 相对初始的变化
            R_delta_cam = self.R_ci @ R_delta_imu @ self.R_ic       # 映射到相机坐标系
            R_imu_in_cam = R_delta_cam.T @ self.camera_initial_rmat # 应用到相机初始姿态（保持与 vis_node 一致）
            q_imu_cam = R.from_matrix(R_imu_in_cam).as_quat().tolist()

            # 互补融合 (IMU 高频, Vision 低频): 越靠近 1 越贴近 IMU
            q_fused = self.slerp(q_vis, q_imu_cam, self.imu_weight)

            # 二次平滑 (与上一帧融合结果做插值)
            if self.last_fused_quat is not None and self.smooth_factor > 0.0:
                q_fused = self.slerp(self.last_fused_quat, q_fused, self.smooth_factor)
            self.last_fused_quat = q_fused

            # 转成旋转向量 rvec
            rot = R.from_quat(q_fused)
            rvec = rot.as_rotvec().reshape(3, 1)  # (3,1) 与 vision 节点保持一致

            rot_imu = R.from_quat(q_imu_cam)
            rvec_imu = rot_imu.as_rotvec().reshape(3, 1)

            # 位姿位置部分直接采用视觉结果
            tvec = np.array([
                vision_pose.pose.position.x,
                vision_pose.pose.position.y,
                vision_pose.pose.position.z
            ], dtype=np.float64).reshape(3, 1)

            stamp = vision_pose.header.stamp  # 使用视觉时间戳
            self.publish_pose(rvec, tvec, stamp)
            self.imu_corrected_pose(rvec_imu, tvec, stamp)
            self.frame_count += 1
            if self.frame_count % 50 == 0:
                self.get_logger().info(f'Fusion frames published: {self.frame_count}')
        except Exception as e:
            self.get_logger().warn(f'Fusion failed: {e}')

    def publish_pose(self, rvec, tvec, stamp):
        """发布姿态 (使用同步后的时间戳)"""
        rotation = R.from_rotvec(rvec.flatten())
        quat = rotation.as_quat()  # [x, y, z, w]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.output_frame

        pose_msg.pose.position.x = float(tvec[0])
        pose_msg.pose.position.y = float(tvec[1])
        pose_msg.pose.position.z = float(tvec[2])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self.pose_pub.publish(pose_msg)

    def imu_corrected_pose(self, rvec, tvec, stamp):
        """发布校正后的 IMU 姿态 (使用同步后的时间戳)"""
        rotation = R.from_rotvec(rvec.flatten())
        quat = rotation.as_quat()  # [x, y, z, w]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.output_frame

        pose_msg.pose.position.x = float(tvec[0])
        pose_msg.pose.position.y = float(tvec[1])
        pose_msg.pose.position.z = float(tvec[2])

        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        self.imu_correct_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FusionPoseNode()
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
