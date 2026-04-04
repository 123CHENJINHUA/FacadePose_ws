#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

class ImuTransNode(Node):
    def __init__(self):
        super().__init__('imu_trans_node')
        
        # 创建与RealSense匹配的QoS配置
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 关键！改成Best Effort
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # 使用自定义QoS订阅
        self.subscription = self.create_subscription(
            Imu,
            '/camera/camera_right/imu',
            self.imu_callback,
            qos_profile)  # 这里传入qos_profile
        
        self.publisher = self.create_publisher(
            Imu,
            'imu_corrected',
            10)  # 发布者可以用默认QoS
        
        self.get_logger().info('IMU坐标系转换节点已启动 (QoS: BEST_EFFORT)')

    def imu_callback(self, msg):
        # 创建新消息，复制原有header等信息
        new_msg = Imu()
        new_msg.header = msg.header
        new_msg.header.frame_id = 'imu_link'  # 可选的，改个清晰的frame_id
        
        # 旋转矩阵：绕X轴旋转-90度（将Y轴重力转到Z轴）
        # 原始坐标系: X右, Y下, Z前 (Realsense默认)
        # 目标坐标系: X右, Y前, Z上 (ROS标准)
        
        # 转换加速度
        acc_x = msg.linear_acceleration.x
        acc_y = msg.linear_acceleration.y
        acc_z = msg.linear_acceleration.z
        
        # 应用旋转: R = [1, 0, 0; 0, 0, 1; 0, -1, 0]
        new_msg.linear_acceleration.x = -acc_x
        new_msg.linear_acceleration.y = -acc_y 
        new_msg.linear_acceleration.z = -acc_z   

        # 同样处理角速度
        gyr_x = msg.angular_velocity.x
        gyr_y = msg.angular_velocity.y
        gyr_z = msg.angular_velocity.z
        
        new_msg.angular_velocity.x = gyr_x
        new_msg.angular_velocity.y = gyr_z
        new_msg.angular_velocity.z = -gyr_y
        
        # 复制协方差矩阵（保持不变）
        new_msg.orientation_covariance = msg.orientation_covariance
        new_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        new_msg.linear_acceleration_covariance = msg.linear_acceleration_covariance
        
        # orientation字段保持为空（VINS不使用）
        new_msg.orientation.x = 0.0
        new_msg.orientation.y = 0.0
        new_msg.orientation.z = 0.0
        new_msg.orientation.w = 1.0
        
        self.publisher.publish(new_msg)
        
        # 打印转换前后的数据对比（可选，调试用）
        self.get_logger().debug(f'原加速度: x={acc_x:.2f}, y={acc_y:.2f}, z={acc_z:.2f}', throttle_duration_sec=1.0)
        self.get_logger().debug(f'新加速度: x={new_msg.linear_acceleration.x:.2f}, '
                               f'y={new_msg.linear_acceleration.y:.2f}, '
                               f'z={new_msg.linear_acceleration.z:.2f}', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ImuTransNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()