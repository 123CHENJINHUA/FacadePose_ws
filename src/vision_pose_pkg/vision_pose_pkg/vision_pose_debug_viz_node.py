#!/usr/bin/env python3
"""vision_pose_debug_viz_node.py

用途
----
把算法每一步的中间结果“可视化并落盘”，方便写文章：
- 原始彩色图/深度伪彩
- 平面分割后的点云投影/mask（尽力而为，不依赖 Open3D 也能跑）
- 直线检测结果叠加
- 最终姿态（rvec/tvec）信息叠加

输入
----
优先复用你现有的在线管线（ros2 bag 播放时同样适用）：
- 与 `vision_pose_node.py` 相同的 depth/color/pointcloud/imu（可选）同步

输出
----
两种方式都支持：
1) 保存逐帧 PNG（推荐写文章用，便于挑“某一时刻”的图）
2) 可选保存为 MP4（用于演示/补充材料）

设计原则
--------
- 不修改你的核心算法实现；尽量复用 `PlaneDetector` 与 `VisionPoseEstimator`
- “文章图”场景下，PNG 序列最灵活：可精确选帧、可无损、后期排版方便

参数
----
- save_dir: 输出目录（默认 ~/.ros/vision_pose_debug）
- save_mode: 'png' | 'video' | 'both'
- save_every_n: 每 N 帧保存一次（减小开销）
- trigger_topic: 触发保存单帧（std_msgs/String）。发布任意字符串即可保存当前帧
- target_stamp_sec: （可选）接近该时间戳（sec）的帧才保存一次（便于定位 bag 中某个时刻）

备注
----
如果你已经在 `vision_pose_node.py` 发布了 `/vision_plane_points` 和 `/vision_detected_line`，
也可以改成本节点只订阅这些“中间结果话题”来做可视化（工作量更小）。
当前实现为“自包含”，直接用原始输入复现每一步，并可保存中间图。
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import QuaternionStamped
from std_msgs.msg import String

from scipy.spatial.transform import Rotation as R

from sensor_msgs_py import point_cloud2

from vision_pose_pkg.plane_detector import PlaneDetector
from vision_pose_pkg.pose_estimator import VisionPoseEstimator


@dataclass
class DebugFrame:
    stamp: rclpy.time.Time
    color_bgr: np.ndarray
    depth: np.ndarray
    cloud_xyz: np.ndarray
    imu_quat_xyzw: Optional[np.ndarray]


class VisionPoseDebugVizNode(Node):
    def __init__(self):
        super().__init__('vision_pose_debug_viz_node')

        # Topics (align with vision_pose_node.py defaults)
        self.declare_parameter('depth_topic', '/camera/camera_middle/aligned_depth_to_color/image_raw')
        self.declare_parameter('color_topic', '/camera/camera_middle/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_middle/color/camera_info')
        self.declare_parameter('pointcloud_topic', '/camera/camera_middle/depth/color/points')
        self.declare_parameter('imu_quat_topic', 'imu/quaternion')

        # Output controls
        self.declare_parameter('save_dir', '/home/cjh/Data_analysis/Facade_vis/video')
        self.declare_parameter('save_mode', 'video')  # png|video|both
        self.declare_parameter('save_every_n', 1)
        self.declare_parameter('video_fps', 30.0)
        self.declare_parameter('video_name', 'debug.mp4')
        self.declare_parameter('trigger_topic', '/vision_pose_debug/trigger_save')
        self.declare_parameter('target_stamp_sec', -1.0)  # if >0, save the closest frame once
        self.declare_parameter('show_window', True)  # 方案A：opencv 窗口显示
        self.declare_parameter('window_name', 'vision_pose_debug')
        self.declare_parameter('window_wait_ms', 1)  # imshow 需要 waitKey
        self.declare_parameter('write_video', True)  # 强制只写视频
        self.declare_parameter('panel_layout', 'h')  # h: 1x3, v: 3x1
        self.declare_parameter('normal_draw_scale', 60.0)  # 法向量箭头长度(像素)
        self.declare_parameter('max_planes_draw', 12)  # 最多画多少个 plane，避免太慢

        # Plane params
        self.declare_parameter('min_plane_points', 300)
        self.declare_parameter('plane_distance_threshold', 0.02)
        self.declare_parameter('voxel_size', 0.02)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.color_topic = self.get_parameter('color_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.imu_quat_topic = self.get_parameter('imu_quat_topic').value

        self.save_dir = self.get_parameter('save_dir').value
        self.save_mode = str(self.get_parameter('save_mode').value).lower()
        self.save_every_n = int(self.get_parameter('save_every_n').value)
        self.video_fps = float(self.get_parameter('video_fps').value)
        self.video_name = self.get_parameter('video_name').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.target_stamp_sec = float(self.get_parameter('target_stamp_sec').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.window_wait_ms = int(self.get_parameter('window_wait_ms').value)

        # FIX: bind parameters to attributes used later
        self.write_video = bool(self.get_parameter('write_video').value)
        self.panel_layout = str(self.get_parameter('panel_layout').value).lower()
        self.normal_draw_scale = float(self.get_parameter('normal_draw_scale').value)
        self.max_planes_draw = int(self.get_parameter('max_planes_draw').value)

        min_points = int(self.get_parameter('min_plane_points').value)
        distance_threshold = float(self.get_parameter('plane_distance_threshold').value)
        voxel_size = float(self.get_parameter('voxel_size').value)

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()

        # Camera intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False

        # TF between camera and imu (copy from vision_pose_node.py)
        self.R_ic = np.array(
            [[0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self.R_ci = self.R_ic.T

        self.plane_detector = PlaneDetector(
            min_points=min_points,
            distance_threshold=distance_threshold,
            voxel_size=voxel_size,
        )
        self.pose_estimator: Optional[VisionPoseEstimator] = None

        # Subscribers
        self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, 10)

        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.color_sub = Subscriber(self, Image, self.color_topic)
        self.cloud_sub = Subscriber(self, PointCloud2, self.pointcloud_topic)
        self.imu_sub = Subscriber(self, QuaternionStamped, self.imu_quat_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.depth_sub, self.color_sub, self.cloud_sub, self.imu_sub],
            queue_size=30,
            slop=0.03,
            allow_headerless=False,
        )
        self.ts.registerCallback(self._on_sync)

        # Manual trigger (save current frame)
        self.trigger_sub = self.create_subscription(String, self.trigger_topic, self._on_trigger, 10)
        self._trigger_requested = False

        # Video writer
        self._video_writer = None
        self._video_size = None

        # State
        self._frame_idx = 0
        self._saved_target_once = False
        self._last_frame: Optional[DebugFrame] = None

        self.get_logger().info('VisionPoseDebugVizNode started')
        self.get_logger().info(f'  save_dir={self.save_dir}')
        self.get_logger().info(f'  save_mode={self.save_mode}, save_every_n={self.save_every_n}')
        self.get_logger().info(f'  trigger_topic={self.trigger_topic}')
        self.get_logger().info(f'  show_window={self.show_window} (window_name={self.window_name})')
        self.get_logger().info(
            f'  write_video={self.write_video}, panel_layout={self.panel_layout}, '
            f'normal_draw_scale={self.normal_draw_scale}, max_planes_draw={self.max_planes_draw}'
        )
        if self.target_stamp_sec > 0:
            self.get_logger().info(f'  target_stamp_sec={self.target_stamp_sec} (will save closest frame once)')

    def _on_camera_info(self, msg: CameraInfo):
        if self.camera_info_received:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.pose_estimator = VisionPoseEstimator(self.camera_matrix, self.dist_coeffs)
        self.camera_info_received = True
        self.get_logger().info('CameraInfo received; pose_estimator initialized')

    def _on_trigger(self, msg: String):
        # Any string triggers a save of current frame
        self._trigger_requested = True

    def _on_sync(self, depth_msg: Image, color_msg: Image, cloud_msg: PointCloud2, imu_msg: QuaternionStamped):
        if not self.camera_info_received or self.pose_estimator is None:
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')

            pts = point_cloud2.read_points_numpy(cloud_msg, field_names=("x", "y", "z"))
            if pts.size == 0:
                return
            mask = np.isfinite(pts).all(axis=1) & (~np.all(pts == 0, axis=1))
            pts = pts[mask]
            if pts.size == 0:
                return

            quat = np.array([
                imu_msg.quaternion.x,
                imu_msg.quaternion.y,
                imu_msg.quaternion.z,
                imu_msg.quaternion.w,
            ], dtype=np.float64)

            stamp = rclpy.time.Time.from_msg(color_msg.header.stamp)
            self._last_frame = DebugFrame(
                stamp=stamp,
                color_bgr=color,
                depth=depth,
                cloud_xyz=pts.astype(np.float32),
                imu_quat_xyzw=quat,
            )

            self._frame_idx += 1
            if self._frame_idx % max(self.save_every_n, 1) != 0:
                # still allow manual trigger / target stamp
                if not self._trigger_requested and not self._should_save_target(stamp):
                    return

            self._process_and_maybe_save(self._last_frame)

        except Exception as e:
            self.get_logger().warn(f'Sync conversion failed: {e}')

    def _should_save_target(self, stamp: rclpy.time.Time) -> bool:
        if self.target_stamp_sec <= 0 or self._saved_target_once:
            return False
        t = stamp.nanoseconds * 1e-9
        # save the first frame that passes the target (good enough for bag playback)
        if t >= self.target_stamp_sec:
            return True
        return False

    def _process_and_maybe_save(self, frame: DebugFrame):
        # 1) 原图
        panel_raw = frame.color_bgr.copy()

        # 2) 平面检测 + select_best_plane 可视化（group 同色，best_plane 红色）
        planes = self.plane_detector.find_planes(frame.cloud_xyz)
        best_plane = None
        plane_dbg = None
        if planes:
            best_plane, plane_dbg = self.plane_detector.select_best_plane(planes, return_debug=True)

        panel_plane = self._make_plane_groups_panel(
            frame.color_bgr.shape[:2],
            planes,
            best_plane,
            plane_dbg,
        )

        # 3) 直线检测结果：只画最终那条线
        panel_line = frame.color_bgr.copy()
        gravity_camera = None
        if frame.imu_quat_xyzw is not None:
            try:
                R_wb = R.from_quat(frame.imu_quat_xyzw).as_matrix()
                R_bw = R_wb.T
                g_world = np.array([0.0, 0.0, -1.0])
                gravity_imu = R_bw @ g_world
                gravity_camera = self.R_ci @ gravity_imu
            except Exception:
                gravity_camera = None

        rvec = tvec = longest_line = None
        if best_plane is not None and self.pose_estimator is not None:
            # 不需要 debug dict；只要最后直线
            rvec, tvec, longest_line = self.pose_estimator.calculate_rotation_angle_from_line(
                best_plane,
                frame.color_bgr,
                gravity_camera,
            )

        if longest_line is not None:
            x1, y1, x2, y2 = [int(v) for v in longest_line]
            cv2.line(panel_line, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # --- NEW: draw gravity projection on image (2D arrow) ---
        try:
            if gravity_camera is not None and np.all(np.isfinite(gravity_camera)):
                g = np.array(gravity_camera, dtype=np.float64).reshape(3)
                n = float(np.linalg.norm(g))
                if n > 1e-9:
                    g = g / n
                    # camera frame: x right, y down. Use (gx, gy) to draw arrow.
                    gx, gy = float(g[0]), float(g[1])
                    h, w = panel_line.shape[:2]
                    # draw at image center
                    p0 = (int(w * 0.5), int(h * 0.5))
                    L = int(min(h, w) * 0.22)
                    p1 = (int(p0[0] + gx * L), int(p0[1] + gy * L))
                    cv2.arrowedLine(panel_line, p0, p1, (0, 0, 255), 3, tipLength=0.25)
                    cv2.putText(panel_line, 'g proj', (p0[0] + 8, p0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(panel_line, 'g proj', (p0[0] + 8, p0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        except Exception:
            pass

        cv2.putText(panel_line, 'Line detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(panel_line, 'Line detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 4) 最终结果：只高亮 best_plane（让人一眼知道选中了哪一块） + 坐标轴
        panel_final = frame.color_bgr.copy()
        if best_plane is not None:
            try:
                # highlight selected plane with stronger alpha and distinct color
                panel_final = self._overlay_plane_on_image(panel_final, best_plane, color=(0, 0, 255), alpha=0.45)
            except Exception:
                pass

        # NOTE: do NOT draw longest_line on panel_final (per request)

        if rvec is not None and tvec is not None and self.camera_matrix is not None and self.dist_coeffs is not None:
            try:
                cv2.drawFrameAxes(
                    panel_final,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec.reshape(3, 1),
                    tvec.reshape(3, 1),
                    length=0.08,
                    thickness=3,
                )
            except Exception:
                pass
        cv2.putText(panel_final, 'Final result', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(panel_final, 'Final result', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 合成 4 个 panel：2x2
        canvas = self._stack_four_panels(panel_raw, panel_plane, panel_line, panel_final)

        # 实时显示
        if self.show_window:
            try:
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(max(self.window_wait_ms, 1)) & 0xFF
                if key == 27:
                    self.get_logger().info('ESC pressed, shutting down...')
                    rclpy.shutdown()
                    return
            except Exception as e:
                self.get_logger().warn(f'cv2.imshow failed (no GUI?): {e}')
                self.show_window = False

        # 只保存视频
        if self.write_video:
            self._write_video(canvas)

    def _stack_four_panels(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
        """2x2 grid; each panel resized to p1 size."""
        h, w = p1.shape[:2]

        def resize(img):
            if img is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            if img.shape[:2] != (h, w):
                return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            return img

        a, b, c, d = resize(p1), resize(p2), resize(p3), resize(p4)
        top = np.hstack([a, b])
        bot = np.hstack([c, d])
        return np.vstack([top, bot])

    def _overlay_plane_on_image(self, image_bgr: np.ndarray, plane: dict, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
        """Project plane 3D points onto image and alpha-blend (vis_node-like).

        This follows the same approach as `vis_pkg/vis_pkg/vis_node.py`:
        - cv2.projectPoints with rvec/tvec = zeros (points are in camera frame)
        - draw circles on an overlay and alpha-blend
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image_bgr

        pts3 = None
        if hasattr(plane.get('pcd', None), 'points'):
            pts3 = np.asarray(plane['pcd'].points)
        elif isinstance(plane.get('pcd', None), np.ndarray):
            pts3 = plane['pcd']
        elif 'points' in plane:
            pts3 = np.asarray(plane['points'])

        if pts3 is None or pts3.size == 0:
            return image_bgr

        pts3 = np.asarray(pts3, dtype=np.float64).reshape(-1, 3)
        valid = np.isfinite(pts3).all(axis=1) & (pts3[:, 2] > 1e-6)
        pts3 = pts3[valid]
        if pts3.size == 0:
            return image_bgr

        try:
            pts2d, _ = cv2.projectPoints(
                pts3,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.camera_matrix,
                self.dist_coeffs,
            )
            pts2d = np.squeeze(pts2d).astype(np.int32)

            if pts2d.ndim == 1:
                pts2d = np.array([pts2d])

            overlay = image_bgr.copy()
            h, w = overlay.shape[:2]

            # draw circles (slightly larger to be visible)
            for x, y in pts2d:
                x = int(x)
                y = int(y)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(overlay, (x, y), 2, color, -1)

            out = cv2.addWeighted(overlay, float(alpha), image_bgr, 1.0 - float(alpha), 0.0)
            return out
        except Exception:
            return image_bgr

    def _write_video(self, frame_bgr: np.ndarray):
        """Lazy-init cv2.VideoWriter and append frame."""
        if frame_bgr is None:
            return

        h, w = frame_bgr.shape[:2]
        if self._video_writer is None:
            os.makedirs(self.save_dir, exist_ok=True)
            out_path = os.path.join(self.save_dir, self.video_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video_writer = cv2.VideoWriter(out_path, fourcc, float(self.video_fps), (w, h))
            self._video_size = (w, h)
            if not self._video_writer.isOpened():
                self.get_logger().error(f'Failed to open VideoWriter: {out_path}')
                self._video_writer = None
                self._video_size = None
                return
            self.get_logger().info(f'VideoWriter opened: {out_path} size={(w, h)} fps={self.video_fps}')

        if self._video_size != (w, h):
            frame_bgr = cv2.resize(frame_bgr, self._video_size, interpolation=cv2.INTER_AREA)

        try:
            self._video_writer.write(frame_bgr)
        except Exception as e:
            self.get_logger().warn(f'VideoWriter.write failed: {e}')

    def destroy_node(self):
        try:
            if self._video_writer is not None:
                self._video_writer.release()
        except Exception:
            pass
        try:
            if self.show_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

    def _make_plane_groups_panel(self, image_hw: Tuple[int, int], planes, best_plane, plane_dbg: Optional[dict]) -> np.ndarray:
        """Render plane groups by projecting each plane points onto image and drawing its normal.

        - same group -> same color
        - best_plane -> red
        """
        h, w = image_hw
        panel = np.zeros((h, w, 3), dtype=np.uint8)

        if not planes or self.camera_matrix is None:
            cv2.putText(panel, 'No planes', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return panel

        # group info: plane index -> group id
        plane_to_gid = {}
        groups = (plane_dbg.get('groups', []) if plane_dbg else [])
        for gid, idxs in enumerate(groups):
            for idx in idxs:
                plane_to_gid[int(idx)] = gid

        # palette (BGR)
        palette = [
            (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255),
            (255, 255, 0), (0, 255, 255), (128, 128, 255), (128, 255, 128),
            (255, 128, 128), (200, 200, 50), (50, 200, 200), (200, 50, 200),
        ]

        best_idx = plane_dbg.get('best_plane_index', None) if plane_dbg else None

        n_draw = min(len(planes), max(self.max_planes_draw, 1))
        for i in range(n_draw):
            p = planes[i]
            gid = plane_to_gid.get(i, i)
            color = palette[gid % len(palette)]
            if best_idx is not None and i == int(best_idx):
                color = (0, 0, 255)

            # get plane points
            pts3 = None
            if hasattr(p.get('pcd', None), 'points'):
                pts3 = np.asarray(p['pcd'].points)
            elif isinstance(p.get('pcd', None), np.ndarray):
                pts3 = p['pcd']
            elif 'points' in p:
                pts3 = np.asarray(p['points'])

            if pts3 is None or pts3.size == 0:
                continue

            valid = np.isfinite(pts3).all(axis=1) & (pts3[:, 2] > 1e-6)
            pts3 = pts3[valid]
            if pts3.size == 0:
                continue

            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            u = (fx * (pts3[:, 0] / pts3[:, 2]) + cx).astype(np.int32)
            v = (fy * (pts3[:, 1] / pts3[:, 2]) + cy).astype(np.int32)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[inside]
            v = v[inside]
            if u.size == 0:
                continue

            panel[v, u] = color

            # draw normal arrow
            center3 = np.array(p.get('center', [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
            normal = np.array(p.get('normal', [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(center3)) or not np.all(np.isfinite(normal)):
                continue
            if center3[2] <= 1e-6:
                continue

            u0 = int(fx * (center3[0] / center3[2]) + cx)
            v0 = int(fy * (center3[1] / center3[2]) + cy)
            if not (0 <= u0 < w and 0 <= v0 < h):
                continue

            tip3 = center3 + 0.15 * normal
            if tip3[2] > 1e-6:
                u1 = int(fx * (tip3[0] / tip3[2]) + cx)
                v1 = int(fy * (tip3[1] / tip3[2]) + cy)
            else:
                u1, v1 = u0, v0

            # fallback to 2D arrow if projection is degenerate
            if abs(u1 - u0) + abs(v1 - v0) < 2:
                u1 = int(u0 + normal[0] * self.normal_draw_scale)
                v1 = int(v0 + normal[1] * self.normal_draw_scale)

            cv2.arrowedLine(panel, (u0, v0), (u1, v1), color, 2, tipLength=0.2)

        panel = cv2.dilate(panel, np.ones((2, 2), np.uint8), iterations=1)
        cv2.putText(panel, 'Group colors; best plane=RED', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(panel, 'Group colors; best plane=RED', (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return panel

def main(args=None):
    rclpy.init(args=args)
    node = VisionPoseDebugVizNode()
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
