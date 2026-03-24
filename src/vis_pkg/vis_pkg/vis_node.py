#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2,CameraInfo
# from nav_msgs.msg import Odometry  # removed: use QuaternionStamped from imu_pose_node
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

# add point cloud helper import
from sensor_msgs_py import point_cloud2

class VisionPoseVizNode(Node):
    def __init__(self):
        super().__init__('vis_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('vision_topic', '/vision_pose')
        self.declare_parameter('fusion_pose_topic', '/fusion_pose')
        self.declare_parameter('output_frame', 'camera_link')
        self.declare_parameter('draw_length', 0.05)
        self.declare_parameter('draw_thickness', 3)
        # use IMU quaternion topic from imu_pose_node
        self.declare_parameter('imu_topic', '/imu_corrected_pose')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        self.image_topic = self.get_parameter('image_topic').value
        self.vision_topic = self.get_parameter('vision_topic').value
        self.fusion_pose_topic = self.get_parameter('fusion_pose_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.draw_length = float(self.get_parameter('draw_length').value)
        self.draw_thickness = int(self.get_parameter('draw_thickness').value)
        self.imu_topic = self.get_parameter('imu_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_stamp = None

        self.latest_vision_pose = None
        self.latest_vision_stamp = None

        self.latest_fusion_pose = None
        self.latest_fusion_stamp = None

        self.latest_imu_pose = None
        self.latest_imu_stamp = None

        # storage for plane points and detected line
        self.plane_points_3d = None  # Nx3 numpy array
        self.longest_line = None     # (x1,y1,x2,y2)
        self.line_type = None

        # Subscriber to image and pose
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(PoseStamped, self.vision_topic, self.vision_pose_callback, 10)
        self.create_subscription(PoseStamped, self.imu_topic, self.imu_pose_callback, 10)
        self.create_subscription(PoseStamped, self.fusion_pose_topic, self.fusion_pose_callback, 10)

        # subscribe to plane point cloud and detected line topics
        self.create_subscription(PointCloud2, '/vision_plane_points', self.plane_pc_callback, 10)
        self.create_subscription(Float32MultiArray, '/vision_detected_line', self.line_callback, 10)

        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.camera_info_received = False

        # Publisher for combined image
        self.combined_image_pub = self.create_publisher(Image, 'combined_image', 10)

        # Timer for visualization
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info('Vision Pose Viz Node started')

    def camera_info_callback(self, msg):
        """相机参数回调"""
        if not self.camera_info_received:
            # 提取相机内参
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.camera_info_received = True



    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = img
            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def imu_pose_callback(self, msg: PoseStamped):
        try:
            q = msg.pose.orientation
            t = msg.pose.position
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            from scipy.spatial.transform import Rotation as R
            Rm = R.from_quat(quat).as_matrix()
            rvec, _ = cv2.Rodrigues(Rm)
            tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
            self.latest_imu_pose = (rvec.flatten(), tvec.flatten())
            self.latest_imu_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert fused pose: {e}')

    def vision_pose_callback(self, msg: PoseStamped):
        # Convert quaternion to rvec and tvec
        try:
            q = msg.pose.orientation
            t = msg.pose.position
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            # Rotation matrix
            from scipy.spatial.transform import Rotation as R
            Rm = R.from_quat(quat).as_matrix()
            rvec, _ = cv2.Rodrigues(Rm)
            tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
            self.latest_vision_pose = (rvec.flatten(), tvec.flatten())
            self.latest_vision_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert pose: {e}')

    def fusion_pose_callback(self, msg: PoseStamped):
        """Convert fused PoseStamped to (rvec, tvec) and store."""
        try:
            q = msg.pose.orientation
            t = msg.pose.position
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            from scipy.spatial.transform import Rotation as R
            Rm = R.from_quat(quat).as_matrix()
            rvec, _ = cv2.Rodrigues(Rm)
            tvec = np.array([t.x, t.y, t.z], dtype=np.float64)
            self.latest_fusion_pose = (rvec.flatten(), tvec.flatten())
            self.latest_fusion_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert fused pose: {e}')

    def plane_pc_callback(self, msg: PointCloud2):
        try:
            # read_points may return generator of tuples, or an ndarray with structured dtype
            points_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = list(points_gen)

            if len(points) == 0:
                self.plane_points_3d = None
                return

            # Handle several possible formats robustly
            if isinstance(points, np.ndarray):
                pts = points
                if pts.dtype.names:  # structured array with fields 'x','y','z'
                    try:
                        arr = np.vstack([pts['x'], pts['y'], pts['z']]).T.astype(np.float32)
                    except Exception:
                        arr = pts.astype(np.float32)
                else:
                    arr = pts.astype(np.float32)
            else:
                # points is a list of tuples/lists -> convert explicitely
                arr = np.array([(float(p[0]), float(p[1]), float(p[2])) for p in points], dtype=np.float32)

            if arr.size == 0:
                self.plane_points_3d = None
            else:
                self.plane_points_3d = arr
        except Exception as e:
            self.get_logger().error(f'Failed to convert plane PointCloud2: {e}')

    def line_callback(self, msg: Float32MultiArray):
        try:
            data = list(msg.data)
            if len(data) >= 4:
                self.longest_line = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
            else:
                self.longest_line = None
        except Exception as e:
            self.get_logger().error(f'Failed to parse detected line: {e}')



    def create_complete_image(self, base_image, rvec, tvec, title=None, title_color=(255,255,255), show_angles=True):
        """Compose a visualization image with plane point overlay, detected longest line, and axes."""
        img = base_image.copy()

        # 1. 绘制点云可视化（将 3D plane 点投影到图像并绘制）
        if self.plane_points_3d is not None and self.plane_points_3d.shape[0] > 0:
            try:
                # project 3D points to image
                pts2d, _ = cv2.projectPoints(self.plane_points_3d, np.zeros(3), np.zeros(3), self.mtx, self.dist)
                pts2d = np.squeeze(pts2d).astype(int)

                overlay = img.copy()
                alpha = 0.5
                color = (255, 0, 0)  # Blue in BGR for plane

                # pts2d may be (N,2) or (2,) for single point
                if pts2d.ndim == 1:
                    pts2d = np.array([pts2d])

                h, w = overlay.shape[:2]
                for pt in pts2d:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(overlay, (x, y), 2, color, -1)

                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            except Exception:
                pass

        # 2. 绘制直线检测结果
        if self.longest_line is not None:
            try:
                x1, y1, x2, y2 = self.longest_line
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow line
                # optionally mark endpoints
                cv2.circle(img, (x1, y1), 4, (0, 0, 255), -1)
                cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)
            except Exception:
                pass

        # 3. 绘制坐标轴
        if rvec is not None and tvec is not None:
            try:
                cv2.drawFrameAxes(img, self.mtx, self.dist, rvec.reshape(-1, 1), tvec.reshape(-1, 1), length=self.draw_length, thickness=self.draw_thickness)
            except Exception:
                pass

        # 4. 标题与线类型
        if title:
            cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2)
        if self.line_type is not None:
            cv2.putText(img, f'Line: {self.line_type}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        return img

    def timer_callback(self):
        if self.latest_image is None:
            return

        base = self.latest_image.copy()

        # Vision panel
        vis_vision = base.copy()
        if self.latest_vision_pose is not None:
            rvec_v, tvec_v = self.latest_vision_pose
            try:
                vis_vision = self.create_complete_image(vis_vision, rvec_v, tvec_v, title='Vision Pose', title_color=(0,255,0))
            except Exception as e:
                self.get_logger().error(f'Failed to compose vision image: {e}')

        # IMU panel
        vis_imu = base.copy()
        if self.latest_imu_pose is not None:
            rvec_i, tvec_i = self.latest_imu_pose
            try:
                vis_imu = self.create_complete_image(vis_imu, rvec_i, tvec_i, title='IMU Pose', title_color=(0,0,255))
            except Exception as e:
                self.get_logger().error(f'Failed to compose IMU image: {e}')
        
        # Fusion panel
        vis_fusion = base.copy()
        if self.latest_fusion_pose is not None:
            rvec_f, tvec_f = self.latest_fusion_pose
            try:
                vis_fusion = self.create_complete_image(vis_fusion, rvec_f, tvec_f, title='Fusion Pose', title_color=(255,0,255))
            except Exception as e:
                self.get_logger().error(f'Failed to compose fusion image: {e}')

        # Combine two panels side-by-side into a single image
        try:
            h1, w1 = vis_vision.shape[:2]
            h2, w2 = vis_imu.shape[:2]
            h3, w3 = vis_fusion.shape[:2]

            target_h = max(h1, h2, h3)

            if h1 != target_h:
                scale = target_h / h1
                vis_vision = cv2.resize(vis_vision, (int(w1 * scale), target_h))
            if h2 != target_h:
                scale = target_h / h2
                vis_imu = cv2.resize(vis_imu, (int(w2 * scale), target_h))
            if h3 != target_h:
                scale = target_h / h3
                vis_fusion = cv2.resize(vis_fusion, (int(w3 * scale), target_h))

            combined = np.hstack((vis_vision, vis_imu, vis_fusion))
            # combined = cv2.resize(combined, (int(combined.shape[1]*0.5), int(combined.shape[0]*0.5)))
        except Exception as e:
            self.get_logger().error(f'Failed to combine images: {e}')
            try:
                combined = np.hstack((vis_vision, vis_imu))
            except Exception:
                combined = vis_vision

        # Publish the combined image
        try:
            combined_msg = self.bridge.cv2_to_imgmsg(combined, encoding="bgr8")
            # Use the timestamp from the latest image if available, otherwise current time
            if self.latest_image_stamp:
                combined_msg.header.stamp = self.latest_image_stamp
            else:
                combined_msg.header.stamp = self.get_clock().now().to_msg()
            combined_msg.header.frame_id = self.output_frame
            self.combined_image_pub.publish(combined_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish combined image: {e}')

        cv2.imshow('Vision+IMU+Fusion Pose', combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VisionPoseVizNode()
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
