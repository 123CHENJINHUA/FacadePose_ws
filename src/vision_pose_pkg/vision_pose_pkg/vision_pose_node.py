#!/usr/bin/env python3
"""
视觉姿态估计节点 (使用 imu_pose_node 发布的 IMU 四元数话题)
订阅RealSense的深度和彩色图像，使用平面检测估计姿态
发布姿态到/vision_pose话题
"""
# import sys
# sys.path.append('/home/cjh/anaconda3/envs/facade_pose_fusion/lib/python3.10/site-packages/')

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from std_msgs.msg import Float32MultiArray, String, Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
# 新增：同步需要
from message_filters import Subscriber, ApproximateTimeSynchronizer
from collections import deque
# 新增：使用里程计的滤波姿态
# from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

# helper to create pointcloud messages
from sensor_msgs_py import point_cloud2

from vision_pose_pkg.plane_detector import PlaneDetector
from vision_pose_pkg.pose_estimator import VisionPoseEstimator



class VisionPoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('vision_pose_estimator_node')
        
        # 参数声明
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('output_topic', '/vision_pose')
        self.declare_parameter('output_frame', 'camera_link')
        self.declare_parameter('publish_rate', 30.0)  # Hz
        # 新增：点云话题（直接订阅 realsense-ros 发布的点云）
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        # 原使用: imu_odom_topic 与 RealSense 原始 IMU, 改为使用 imu_pose_node 发布的四元数
        self.declare_parameter('imu_quat_topic', 'imu/quaternion')
        
        # 获取参数
        self.depth_topic = self.get_parameter('depth_topic').value
        self.color_topic = self.get_parameter('color_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.imu_quat_topic = self.get_parameter('imu_quat_topic').value
        
        # 平面检测参数
        self.declare_parameter('min_plane_points', 300)
        self.declare_parameter('plane_distance_threshold', 0.02)
        self.declare_parameter('voxel_size', 0.02)
        
        min_points = self.get_parameter('min_plane_points').value
        distance_threshold = self.get_parameter('plane_distance_threshold').value
        voxel_size = self.get_parameter('voxel_size').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # 图像缓存
        self.latest_depth = None
        self.latest_color = None
        self.depth_timestamp = None
        self.color_timestamp = None
        # 新增：点云缓存
        self.latest_cloud = None
        # 新增：同步后的统一时间戳
        self.synced_stamp = None
        
        #imu
        self.imu_timestamp = None
        self.latest_imu = None
        
        # camera -> imu
        self.R_ic = np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=np.float64)

        # imu -> camera (transpose/inverse)
        self.R_ci = self.R_ic.T
        
        # 平面检测器和姿态估计器
        self.plane_detector = PlaneDetector(
            min_points=min_points,
            distance_threshold=distance_threshold,
            voxel_size=voxel_size
        )
        self.pose_estimator = None  # 等待相机参数后初始化
        
        # 订阅者（CameraInfo 单独）
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        
        # 使用 message_filters 同步 depth / color / pointcloud
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.color_sub = Subscriber(self, Image, self.color_topic)
        self.pointcloud_sub = Subscriber(self, PointCloud2, self.pointcloud_topic)
        self.imu_sub = Subscriber(self, QuaternionStamped, self.imu_quat_topic)
        # 同步器：允许少量时间偏差 (slop)，根据需要调小/调大
        self.ts = ApproximateTimeSynchronizer(
            [self.depth_sub, self.color_sub, self.pointcloud_sub, self.imu_sub],
            queue_size=15,
            slop=0.03,
            allow_headerless=False
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info('Waiting for synchronized depth/color/pointcloud...')
        
        # 发布姿态
        self.pose_pub = self.create_publisher(PoseStamped,self.output_topic,10)

        # 发布平面点云和直线检测结果
        self.plane_pc_pub = self.create_publisher(PointCloud2, '/vision_plane_points', 10)
        self.line_pub = self.create_publisher(Float32MultiArray, '/vision_detected_line', 10)
        
        # 定时器
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_and_publish
        )
        
        # 统计信息
        self.frame_count = 0
        self.success_count = 0
        self.last_log_time = time.time()
        # 记录上一帧可用的角度
        self.prev_angle = None
        # 新增：FPS 统计
        self.start_time = time.time()
        self.last_frame_count = 0
        self.last_success_count = 0
        
        self.get_logger().info('Vision Pose Estimator Node started')
        self.get_logger().info(f'  Depth topic: {self.depth_topic}')
        self.get_logger().info(f'  Color topic: {self.color_topic}')
        self.get_logger().info(f'  Output topic: {self.output_topic}')
        self.get_logger().info(f'  Publish rate: {self.publish_rate} Hz')
        self.get_logger().info(f'  PointCloud topic: {self.pointcloud_topic}')
        self.get_logger().info('  Using ApproximateTimeSynchronizer for depth/color/pointcloud')
        self.get_logger().info(f'  Using IMU quaternion topic: {self.imu_quat_topic}')
        self.get_logger().info(f'Subscribed to IMU quaternion topic: {self.imu_quat_topic}')
        
    def camera_info_callback(self, msg):
        """相机参数回调"""
        if not self.camera_info_received:
            # 提取相机内参
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            
            # 初始化姿态估计器
            self.pose_estimator = VisionPoseEstimator(
                self.camera_matrix,
                self.dist_coeffs
            )
            
            self.camera_info_received = True
            self.get_logger().info('Camera parameters received')
            self.get_logger().info(f'  fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}')
            self.get_logger().info(f'  cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}')
    
    # 同步回调：接收时间对齐的 depth / color / pointcloud
    def sync_callback(self, depth_msg: Image, color_msg: Image, cloud_msg: PointCloud2, imu_msg: QuaternionStamped):
        if not self.camera_info_received:
            return
        try:
            # 统一使用 color 图的时间戳（RealSense 中二者应接近）
            self.synced_stamp = color_msg.header.stamp
            # 转换图像
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            self.depth_timestamp = depth_msg.header.stamp
            self.color_timestamp = color_msg.header.stamp
            # 点云转换
            pts = point_cloud2.read_points_numpy(cloud_msg, field_names=("x", "y", "z"))
            if pts.size == 0:
                return
            mask = np.isfinite(pts).all(axis=1) & (~np.all(pts == 0, axis=1))
            pts = pts[mask]
            if pts.size == 0:
                return
            self.latest_cloud = pts.astype(np.float32)
            # IMU 四元数回调
            self.imu_timestamp = imu_msg.header.stamp
            quat = [
                imu_msg.quaternion.x,
                imu_msg.quaternion.y,
                imu_msg.quaternion.z,
                imu_msg.quaternion.w
            ]
            self.latest_imu = quat
        except Exception as e:
            self.get_logger().warn(f'Sync callback conversion failed: {e}')
    


    def process_and_publish(self):
        """处理同步后的数据并发布姿态"""
        
        if not self.camera_info_received:
            self.get_logger().info('No camera info yet, cannot process data')
            return
        # 需要同步后的彩色图 + 点云
        if self.latest_color is None or self.latest_cloud is None or self.synced_stamp is None:
            # 提示尚未同步到三路消息
            # self.get_logger().debug('Waiting for synchronized messages...')
            self.get_logger().info('Waiting for synchronized messages...')
            return
        
        self.frame_count += 1
        # self.get_logger().info('Start processing ...')

        
        try:
            point_cloud = self.latest_cloud
            if point_cloud is None or len(point_cloud) < 100:
                return
            
            # 检测平面
            planes = self.plane_detector.find_planes(point_cloud)
            if not planes:
                return
            
            best_plane = self.plane_detector.select_best_plane(planes)
            if best_plane is None:
                return
            
            # 使用 IMU 四元数获取重力方向
            gravity_camera = None
            quat = self.latest_imu
            if quat is not None:
                R_wb = R.from_quat(quat).as_matrix()
                R_bw = R_wb.T
                g_world = np.array([0.0, 0.0, -1.0])
                gravity_imu = R_bw @ g_world
                gravity_camera = self.R_ci @ gravity_imu
            else:
                self.get_logger().debug('No matching IMU quaternion, skip gravity correction this frame')
            
            rvec, tvec, longest_line = self.pose_estimator.calculate_rotation_angle_from_line(
                best_plane,
                self.latest_color,
                gravity_camera
            )
            
            
            if rvec is None or tvec is None:
                self.get_logger().warn('Failed to estimate pose from plane')
                return
            self.get_logger().debug('Estimated pose from plane')
            
            # 只有当存在可用角度或线时再进入发布逻辑
            if longest_line is not None:
                # 发布直线（仅当有线，且用于发布的角度有效时，避免发布 NaN/None）
                arr = Float32MultiArray()
                arr.data = [
                    float(longest_line[0]), float(longest_line[1]),
                    float(longest_line[2]), float(longest_line[3]),
                ]
                self.line_pub.publish(arr)
                self.get_logger().debug('Published line info')

            else:
                self.get_logger().warn('Failed to detect line features')
            
            # 发布姿态 (使用同步时间戳)
            self.publish_pose(rvec, tvec, self.synced_stamp)
            self.get_logger().debug('Published pose')
            
            # 发布平面点云
            plane_points_3d = None
            try:
                if isinstance(best_plane.get('pcd', None), np.ndarray):
                    plane_points_3d = best_plane['pcd']
                elif hasattr(best_plane.get('pcd', None), 'points'):
                    plane_points_3d = np.asarray(best_plane['pcd'].points)
                elif 'points' in best_plane:
                    plane_points_3d = np.array(best_plane['points'])
            except Exception:
                plane_points_3d = None
            
            if plane_points_3d is not None and plane_points_3d.size > 0:
                header = Header()
                header.stamp = self.synced_stamp
                header.frame_id = self.output_frame
                try:
                    cloud_msg = point_cloud2.create_cloud_xyz32(header, plane_points_3d.tolist())
                    self.plane_pc_pub.publish(cloud_msg)
                except Exception as e:
                    self.get_logger().warn(f'Failed to publish plane pointcloud: {e}')
            
            self.success_count += 1
            current_time = time.time()
            if current_time - self.last_log_time > 5.0:
                success_rate = (self.success_count / self.frame_count * 100) if self.frame_count > 0 else 0
                # 5秒窗口 FPS 与平均 FPS
                frames_delta = self.frame_count - self.last_frame_count
                success_delta = self.success_count - self.last_success_count
                elapsed = current_time - self.last_log_time
                fps = (frames_delta / elapsed) if elapsed > 0 else 0.0
                success_fps = (success_delta / elapsed) if elapsed > 0 else 0.0
                avg_fps = self.frame_count / max((current_time - self.start_time), 1e-6)
                self.get_logger().info(
                    f'Processed {self.frame_count} frames, success: {self.success_count} ({success_rate:.1f}%), '
                    f'fps: {fps:.1f} (succ {success_fps:.1f}), avg: {avg_fps:.1f}'
                )
                self.last_log_time = current_time
                self.last_frame_count = self.frame_count
                self.last_success_count = self.success_count
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
    
    def publish_pose(self, rvec, tvec, stamp):
        """发布姿态 (使用同步后的时间戳)"""
        from scipy.spatial.transform import Rotation as R
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


def main(args=None):
    rclpy.init(args=args)
    
    node = VisionPoseEstimatorNode()
    
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
