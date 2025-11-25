#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较节点: 订阅 /vision_pose, /imu_corrected_pose, /fusion_pose
记录姿态 (四元数 -> 欧拉角) 并定期绘制对比图保存到目录.
"""
import os
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from dataclasses import dataclass

try:
    from scipy.spatial.transform import Rotation as R
    _has_scipy = True
except Exception:
    _has_scipy = False

try:
    import matplotlib
    matplotlib.use('Agg')  # 后端使用非交互以便保存
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except Exception:
    _has_matplotlib = False

@dataclass
class PoseRecord:
    t: float                # 相对开始的时间 (秒)
    stamp: float            # ROS 时间戳 (秒)
    quat: np.ndarray        # 归一化四元数 [x,y,z,w]
    euler: np.ndarray       # 欧拉角 [roll, pitch, yaw] (deg)

class PoseComparisonNode(Node):
    def __init__(self):
        super().__init__('pose_comparison_node')

        # 参数
        self.declare_parameter('vision_topic', '/vision_pose')
        self.declare_parameter('imu_topic', '/imu_corrected_pose')
        self.declare_parameter('fusion_topic', '/fusion_pose')
        self.declare_parameter('save_period', 5.0)            # 每多少秒保存一次图
        # 修改默认保存目录: 放到包内 comparison_pkg/comparison_pkg/comparison
        self.declare_parameter('output_dir', 'comparison')
        self.declare_parameter('euler_order', 'xyz')          # 欧拉角顺序 (scipy 规范)
        self.declare_parameter('degrees', True)               # 是否用角度
        self.declare_parameter('max_points', 10000)           # 限制最多存储点数

        self.vision_topic = self.get_parameter('vision_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.fusion_topic = self.get_parameter('fusion_topic').value
        self.save_period = float(self.get_parameter('save_period').value)
        self.output_dir = self.get_parameter('output_dir').value
        self.euler_order = self.get_parameter('euler_order').value
        self.use_degrees = bool(self.get_parameter('degrees').value)
        self.max_points = int(self.get_parameter('max_points').value)

        # 解析输出目录: 若在安装后的 site-packages 中运行, 自动映射到源码 src/comparison_pkg/comparison_pkg
        self.output_dir = self._resolve_output_dir(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # 数据缓存
        self.start_time = time.time()
        self.records = {
            'vision': [],
            'imu': [],
            'fusion': []
        }

        # 订阅
        self.create_subscription(PoseStamped, self.vision_topic, lambda msg: self.pose_callback(msg, 'vision'), 50)
        self.create_subscription(PoseStamped, self.imu_topic, lambda msg: self.pose_callback(msg, 'imu'), 50)
        self.create_subscription(PoseStamped, self.fusion_topic, lambda msg: self.pose_callback(msg, 'fusion'), 50)

        # 定时保存
        self.last_save_time = time.time()
        self.timer = self.create_timer(0.5, self.timer_callback)  # 2Hz 检查

        self.get_logger().info('PoseComparisonNode started')
        self.get_logger().info(f' vision_topic: {self.vision_topic}')
        self.get_logger().info(f' imu_topic: {self.imu_topic}')
        self.get_logger().info(f' fusion_topic: {self.fusion_topic}')
        self.get_logger().info(f' output_dir(resolved): {self.output_dir}')
        if not _has_matplotlib:
            self.get_logger().warn('matplotlib 不可用，无法绘图保存')
        if not _has_scipy:
            self.get_logger().warn('scipy 不可用，无法进行四元数到欧拉角转换')

        # 移除 add_on_shutdown (rclpy Node 不支持), 在 main() 中手动调用 on_shutdown
        # self.add_on_shutdown(self.on_shutdown)

    def _norm_quat(self, q):
        q = np.array(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.,0.,0.,1.], dtype=np.float64)
        return q / n

    def _quat_to_euler(self, q):
        if not _has_scipy:
            return np.zeros(3, dtype=np.float64)
        rot = R.from_quat(q)
        return rot.as_euler(self.euler_order, degrees=self.use_degrees)

    def pose_callback(self, msg: PoseStamped, key: str):
        try:
            q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
            q = self._norm_quat(q)
            euler = self._quat_to_euler(q)
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            t_rel = time.time() - self.start_time
            rec = PoseRecord(t=t_rel, stamp=stamp_sec, quat=q, euler=euler)
            arr = self.records[key]
            arr.append(rec)
            if len(arr) > self.max_points:
                # 简单下采样: 保留后半部分
                self.records[key] = arr[-self.max_points//2:]
        except Exception as e:
            self.get_logger().warn(f'Failed to process pose for {key}: {e}')

    def timer_callback(self):
        if not _has_matplotlib or not _has_scipy:
            return
        now = time.time()
        if now - self.last_save_time >= self.save_period:
            self.save_plots()
            self.last_save_time = now

    def _quat_angle_deg(self, q1, q2):
        q1 = self._norm_quat(q1)
        q2 = self._norm_quat(q2)
        dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
        angle = 2.0 * np.arccos(dot)
        return np.degrees(angle) if self.use_degrees else angle

    def save_plots(self):
        try:
            vision = self.records['vision']
            imu = self.records['imu']
            fusion = self.records['fusion']
            if len(fusion) == 0 or (len(vision) == 0 and len(imu) == 0):
                return

            # 准备数据
            def to_arrays(recs):
                t = np.array([r.t for r in recs])
                e = np.vstack([r.euler for r in recs]) if len(recs) else np.zeros((0,3))
                q = np.vstack([r.quat for r in recs]) if len(recs) else np.zeros((0,4))
                return t, e, q
            t_v, e_v, q_v = to_arrays(vision)
            t_i, e_i, q_i = to_arrays(imu)
            t_f, e_f, q_f = to_arrays(fusion)

            # 图 1: 欧拉角对比
            plt.figure(figsize=(10, 6))
            labels = ['Roll', 'Pitch', 'Yaw'] if self.use_degrees else ['Roll(rad)', 'Pitch(rad)', 'Yaw(rad)']
            if len(e_v):
                plt.plot(t_v, e_v[:,0], 'g-', alpha=0.8, label=f'Vision {labels[0]}')
                plt.plot(t_v, e_v[:,1], 'g--', alpha=0.8, label=f'Vision {labels[1]}')
                plt.plot(t_v, e_v[:,2], 'g:', alpha=0.8, label=f'Vision {labels[2]}')
            if len(e_i):
                plt.plot(t_i, e_i[:,0], 'b-', alpha=0.6, label=f'IMU {labels[0]}')
                plt.plot(t_i, e_i[:,1], 'b--', alpha=0.6, label=f'IMU {labels[1]}')
                plt.plot(t_i, e_i[:,2], 'b:', alpha=0.6, label=f'IMU {labels[2]}')
            if len(e_f):
                plt.plot(t_f, e_f[:,0], 'm-', alpha=0.9, label=f'Fusion {labels[0]}')
                plt.plot(t_f, e_f[:,1], 'm--', alpha=0.9, label=f'Fusion {labels[1]}')
                plt.plot(t_f, e_f[:,2], 'm:', alpha=0.9, label=f'Fusion {labels[2]}')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (deg)' if self.use_degrees else 'Angle (rad)')
            plt.title('Euler Angles Comparison')
            plt.legend(ncol=3, fontsize=8)
            plt.grid(alpha=0.3)
            euler_path = os.path.join(self.output_dir, 'euler_comparison.png')
            plt.tight_layout()
            plt.savefig(euler_path, dpi=150)
            plt.close()

            # 图 2: 旋转差异角度 (Fusion vs Vision / IMU)
            if len(q_f):
                angles_f_v = []
                angles_f_i = []
                # 简单使用最近时间匹配 (可以改进为插值)
                def nearest(t_array, t):
                    if len(t_array) == 0:
                        return None
                    idx = int(np.argmin(np.abs(t_array - t)))
                    return idx
                for tf, qf in zip(t_f, q_f):
                    idx_v = nearest(t_v, tf)
                    idx_i = nearest(t_i, tf)
                    if idx_v is not None:
                        angles_f_v.append(self._quat_angle_deg(qf, q_v[idx_v]))
                    if idx_i is not None:
                        angles_f_i.append(self._quat_angle_deg(qf, q_i[idx_i]))
                tf_plot = t_f[:len(angles_f_v)] if len(angles_f_v) else []
                tf_plot_i = t_f[:len(angles_f_i)] if len(angles_f_i) else []
                if len(angles_f_v) or len(angles_f_i):
                    plt.figure(figsize=(8,5))
                    if len(angles_f_v):
                        plt.plot(tf_plot, angles_f_v, 'r-', label='Fusion-Vision Angle Diff')
                    if len(angles_f_i):
                        plt.plot(tf_plot_i, angles_f_i, 'c-', label='Fusion-IMU Angle Diff')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Angle Diff (deg)' if self.use_degrees else 'Angle Diff (rad)')
                    plt.title('Fusion Orientation Difference')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    diff_path = os.path.join(self.output_dir, 'fusion_diff.png')
                    plt.tight_layout()
                    plt.savefig(diff_path, dpi=150)
                    plt.close()

            # 保存一次原始数据 numpy
            np.savez(os.path.join(self.output_dir, 'pose_records.npz'),
                     t_v=t_v, e_v=e_v, q_v=q_v,
                     t_i=t_i, e_i=e_i, q_i=q_i,
                     t_f=t_f, e_f=e_f, q_f=q_f)

            self.get_logger().info(f'保存对比图到 {self.output_dir}')
        except Exception as e:
            self.get_logger().error(f'保存图失败: {e}')

    def on_shutdown(self):
        if _has_matplotlib and _has_scipy:
            self.save_plots()
        self.get_logger().info('PoseComparisonNode shutdown')

    def _resolve_output_dir(self, out_dir):
        # 若用户给绝对路径直接用
        if os.path.isabs(out_dir):
            return out_dir
        base_dir = os.path.dirname(__file__)
        # 默认先按当前文件所在目录
        resolved = os.path.join(base_dir, out_dir)
        # 如果当前文件路径包含 site-packages 说明是安装后的路径, 尝试映射到工作空间 src
        if 'site-packages' in base_dir:
            # 向上找 install 目录
            probe = base_dir
            while probe != '/' and os.path.basename(probe) != 'install':
                probe = os.path.dirname(probe)
            if os.path.basename(probe) == 'install':
                workspace_root = os.path.dirname(probe)
                src_pkg_dir = os.path.join(workspace_root, 'src', 'comparison_pkg', 'comparison_pkg')
                alt = os.path.join(src_pkg_dir, out_dir)
                if os.path.isdir(src_pkg_dir):
                    resolved = alt
        return resolved


def main(args=None):
    rclpy.init(args=args)
    node = PoseComparisonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 手动调用关闭处理
        node.on_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
