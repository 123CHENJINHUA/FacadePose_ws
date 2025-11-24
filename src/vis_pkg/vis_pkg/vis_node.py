#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
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
        self.declare_parameter('pose_topic', '/vision_pose')
        self.declare_parameter('output_frame', 'camera_link')
        self.declare_parameter('draw_length', 0.05)
        self.declare_parameter('draw_thickness', 3)
        # use IMU quaternion topic from imu_pose_node
        self.declare_parameter('imu_quat_topic', 'imu/quaternion')

        self.image_topic = self.get_parameter('image_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.output_frame = self.get_parameter('output_frame').value
        self.draw_length = float(self.get_parameter('draw_length').value)
        self.draw_thickness = int(self.get_parameter('draw_thickness').value)
        self.imu_quat_topic = self.get_parameter('imu_quat_topic').value

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_stamp = None

        self.latest_pose = None
        self.latest_pose_stamp = None

        # storage for plane points and detected line
        self.plane_points_3d = None  # Nx3 numpy array
        self.longest_line = None     # (x1,y1,x2,y2)
        self.line_type = None

        # Subscriber to image and pose
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)

        # subscribe to plane point cloud and detected line topics
        self.create_subscription(PointCloud2, '/vision_plane_points', self.plane_pc_callback, 10)
        self.create_subscription(Float32MultiArray, '/vision_detected_line', self.line_callback, 10)

        # Camera intrinsics will be provided via parameters (for simplicity)
        # Users should set these params to match camera_info
        self.declare_parameter('fx', 525.0)
        self.declare_parameter('fy', 525.0)
        self.declare_parameter('cx', 319.5)
        self.declare_parameter('cy', 239.5)
        self.declare_parameter('dist_coeffs', [])

        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        dist = self.get_parameter('dist_coeffs').value

        self.mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        if dist is None or len(dist) == 0:
            self.dist = np.zeros((5, 1), dtype=np.float32)
        else:
            self.dist = np.array(dist, dtype=np.float32)

        # IMU-camera extrinsics
        self.setup_imu_camera_transform()

        # IMU-related state for IMU-only pose visualization
        self.imu_only_rvec = None
        self.imu_only_tvec = None
        self.imu_initialized = False
        self.last_imu_time = None

        # Store initial rotations for camera and IMU to compute changes
        self.camera_initial_rmat = None   # 3x3 rotation matrix of first camera pose
        self.imu_initial_rmat = None      # 3x3 rotation matrix of first imu quaternion
        self.camera_initial_rvec = None   # optional store initial rvec
        self.camera_initial_tvec = None

        # Timer for visualization
        self.timer = self.create_timer(0.033, self.timer_callback)

        # Subscribe to IMU quaternion topic (from imu_pose_node)
        try:
            self.create_subscription(QuaternionStamped, self.imu_quat_topic, self.imu_quat_callback, 10)
            self.get_logger().info(f'IMU quaternion topic: {self.imu_quat_topic}')
        except Exception:
            self.get_logger().warn('Failed to subscribe to IMU quaternion topic; IMU visualization disabled')

        self.get_logger().info('Vision Pose Viz Node started')

    def setup_imu_camera_transform(self):
        # camera -> imu
        self.R_ic = np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=np.float64)

        # imu -> camera (transpose/inverse)
        self.R_ci = self.R_ic.T

    def imu_quat_callback(self, msg: QuaternionStamped):
        try:
            q = msg.quaternion
            quat_imu = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            from scipy.spatial.transform import Rotation as R

            # Current IMU rotation matrix
            R_imu_curr = R.from_quat(quat_imu).as_matrix()

            # Store initial IMU rotation on first message
            if self.imu_initial_rmat is None:
                self.imu_initial_rmat = R_imu_curr.copy()

            # If camera initial not available yet, keep imu_only as current imu (converted to rvec)
            if self.camera_initial_rmat is None:
                rvec_curr, _ = cv2.Rodrigues(R_imu_curr)
                self.imu_only_rvec = rvec_curr.flatten()
                return

            # Rotation delta in IMU frame 这是基于原坐标系的变化，所以是右乘
            R_delta_imu = self.imu_initial_rmat.T @ R_imu_curr

            # 转欧拉角并输出 (XYZ 依次为 roll, pitch, yaw)
            # euler_deg = R.from_matrix(R_delta_imu).as_euler('xyz', degrees=True)
            # roll, pitch, yaw = euler_deg
            # self.get_logger().info(f'IMU delta euler (deg): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}')

            # Map delta into camera frame using extrinsics

            # 定义角度
            # angle = -45.0 * np.pi / 180.0  # 45 degrees in radians
            # R_delta_imu = np.array([
            #     [np.cos(angle), 0, -np.sin(angle)],
            #     [0, 1, 0],
            #     [np.sin(angle), 0, np.cos(angle)]
            # ], dtype=np.float64)

            R_delta_cam = self.R_ci @ R_delta_imu @ self.R_ic

            # Apply delta to camera initial rotation
            R_final_cam = R_delta_cam.T @ self.camera_initial_rmat

            # Convert to rvec for visualization
            rvec_final, _ = cv2.Rodrigues(R_final_cam)
            self.imu_only_rvec = rvec_final.flatten()

            # ensure imu tvec matches latest vision tvec if available
            if self.latest_pose is not None:
                _, vision_t = self.latest_pose
                self.imu_only_tvec = vision_t.copy()

            self.imu_initialized = True
        except Exception as e:
            self.get_logger().error(f'Failed to convert IMU quaternion to rvec: {e}')

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = img
            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def pose_callback(self, msg: PoseStamped):
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
            self.latest_pose = (rvec.flatten(), tvec.flatten())
            self.latest_pose_stamp = msg.header.stamp

            # On first vision pose, record camera initial rotation and tvec
            if self.camera_initial_rmat is None:
                self.camera_initial_rmat = Rm.copy()
                self.camera_initial_rvec = rvec.flatten().copy()
                self.camera_initial_tvec = tvec.flatten().copy()

            # initialize IMU-only pose if not yet and imu_initial exists
            if not self.imu_initialized and self.imu_initial_rmat is not None:
                # compute imu-only initial as camera initial (apply zero delta)
                R_final = self.camera_initial_rmat.copy()
                rvec_final, _ = cv2.Rodrigues(R_final)
                self.imu_only_rvec = rvec_final.flatten().copy()
                self.imu_only_tvec = tvec.flatten().copy()
                self.imu_initialized = True
                self.last_imu_time = None
        except Exception as e:
            self.get_logger().error(f'Failed to convert pose: {e}')

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
        if self.latest_pose is not None:
            rvec_v, tvec_v = self.latest_pose
            try:
                vis_vision = self.create_complete_image(vis_vision, rvec_v, tvec_v, title='Vision Pose', title_color=(0,255,0))
            except Exception as e:
                self.get_logger().error(f'Failed to compose vision image: {e}')

        # IMU panel
        vis_imu = base.copy()
        if self.imu_initialized and self.imu_only_rvec is not None and self.imu_only_tvec is not None:
            try:
                vis_imu = self.create_complete_image(vis_imu, self.imu_only_rvec, self.imu_only_tvec, title='IMU Pose', title_color=(0,128,255))
            except Exception as e:
                self.get_logger().error(f'Failed to compose imu image: {e}')
        else:
            # show vision pose in imu panel if imu not initialized
            if self.latest_pose is not None:
                try:
                    vis_imu = self.create_complete_image(vis_imu, *self.latest_pose, title='IMU Pose (init)', title_color=(0,128,255))
                except Exception:
                    pass

        # Combine two panels side-by-side into a single image
        try:
            h1, w1 = vis_vision.shape[:2]
            h2, w2 = vis_imu.shape[:2]
            target_h = max(h1, h2)

            if h1 != target_h:
                scale = target_h / h1
                vis_vision = cv2.resize(vis_vision, (int(w1 * scale), target_h))
            if h2 != target_h:
                scale = target_h / h2
                vis_imu = cv2.resize(vis_imu, (int(w2 * scale), target_h))

            combined = np.hstack((vis_vision, vis_imu))
            # combined = cv2.resize(combined, (int(combined.shape[1]*0.5), int(combined.shape[0]*0.5)))
        except Exception as e:
            self.get_logger().error(f'Failed to combine images: {e}')
            combined = vis_vision

        cv2.imshow('Vision+IMU Pose', combined)
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
