#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import struct
import time
import serial
import serial.tools.list_ports
import transforms3d as tfs

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, QuaternionStamped, Vector3Stamped
from tf2_ros import TransformBroadcaster

# 协议常量
FRAME_HEAD = 'fc'
FRAME_END = 'fd'
TYPE_IMU = '40'
TYPE_AHRS = '41'
TYPE_INSGPS = '42'
TYPE_GEODETIC_POS = '5c'
TYPE_GROUND = 'f0'
TYPE_SYS_STATE = '50'
TYPE_BODY_ACCELERATION = '62'
TYPE_ACCELERATION = '61'
TYPE_MSG_BODY_VEL = '60'

IMU_LEN = '38'              # 56 bytes
AHRS_LEN = '30'             # 48 bytes
INSGPS_LEN = '48'           # 72 bytes
GEODETIC_POS_LEN = '20'     # 32 bytes
SYS_STATE_LEN = '64'        # 100 bytes
BODY_ACCELERATION_LEN = '10'# 16 bytes
ACCELERATION_LEN = '0c'     # 12 bytes

PI = 3.141592653589793
RAD2DEG = 180.0 / PI


class ImuPoseNode(Node):
    def __init__(self) -> None:
        super().__init__('imu_pose_node')

        # 参数
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('timeout_ms', 20)  # 串口超时(ms)
        self.declare_parameter('parent_frame_id', 'map')
        self.declare_parameter('child_frame_id', 'imu_link')

        self.port: str = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate: int = self.get_parameter('baudrate').get_parameter_value().integer_value
        timeout_ms: int = self.get_parameter('timeout_ms').get_parameter_value().integer_value
        self.timeout = timeout_ms
        self.parent_frame_id: str = self.get_parameter('parent_frame_id').get_parameter_value().string_value
        self.child_frame_id: str = self.get_parameter('child_frame_id').get_parameter_value().string_value

        # 发布者与TF
        self.q_pub = self.create_publisher(QuaternionStamped, 'imu/quaternion', 10)
        self.euler_rad_pub = self.create_publisher(Vector3Stamped, 'imu/euler_rad', 10)
        self.euler_deg_pub = self.create_publisher(Vector3Stamped, 'imu/euler_deg', 10)
        self.gyro_pub = self.create_publisher(Vector3Stamped, 'imu/gyro', 10)
        self.acc_pub = self.create_publisher(Vector3Stamped, 'imu/acceleration', 10)
        self.mag_pub = self.create_publisher(Vector3Stamped, 'imu/magnetic', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 串口
        self.serial_ = None
        self.stop_event = threading.Event()
        self.rx_thread = threading.Thread(target=self._receive_loop, daemon=True)

        # 尝试打开串口并启动线程
        if self._open_port():
            self.rx_thread.start()
        else:
            self.get_logger().error('未能打开串口，节点仍在运行，可修改参数后重启。')


    def destroy_node(self):
        # 优雅关闭
        self.stop_event.set()
        try:
            if self.rx_thread.is_alive():
                self.rx_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.serial_ and self.serial_.is_open:
                self.serial_.close()
        except Exception:
            pass
        return super().destroy_node()

    # 串口工具
    def _find_port_exists(self) -> bool:
        for p in serial.tools.list_ports.comports():
            if p.device == self.port:
                return True
        return False

    def _open_port(self) -> bool:
        if not self._find_port_exists():
            self.get_logger().error(f'找不到串口: {self.port}')
            return False
        try:
            self.serial_ = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
            )
            self.get_logger().info(f'串口已打开: {self.serial_.port} 波特率: {self.serial_.baudrate}')
            return True
        except Exception as e:
            self.get_logger().error(f'打开串口失败: {e}')
            return False

    # 读取循环
    def _receive_loop(self):
        ser = self.serial_
        if ser is None:
            return
        while (not self.stop_event.is_set()) and ser.is_open:
            try:
                if not threading.main_thread().is_alive():
                    print('done')
                    break
                check_head = ser.read().hex()
                # 校验帧头
                if check_head != FRAME_HEAD:
                    continue
                head_type = ser.read().hex()
                # 校验数据类型
                if (head_type != TYPE_IMU and head_type != TYPE_AHRS and head_type != TYPE_INSGPS and
                        head_type != TYPE_GEODETIC_POS and head_type != 0x50 and head_type != TYPE_GROUND and 
                        head_type != TYPE_SYS_STATE and  head_type!=TYPE_MSG_BODY_VEL and head_type!=TYPE_BODY_ACCELERATION and head_type!=TYPE_ACCELERATION):
                    continue
                check_len = ser.read().hex()
                # 校验数据类型的长度
                if head_type == TYPE_IMU and check_len != IMU_LEN:
                    continue
                elif head_type == TYPE_AHRS and check_len != AHRS_LEN:
                    continue
                elif head_type == TYPE_INSGPS and check_len != INSGPS_LEN:
                    continue
                elif head_type == TYPE_GEODETIC_POS and check_len != GEODETIC_POS_LEN:
                    continue
                elif head_type == TYPE_SYS_STATE and check_len != SYS_STATE_LEN:
                    continue
                elif head_type == TYPE_GROUND or head_type == 0x50:
                    continue
                elif head_type == TYPE_MSG_BODY_VEL and check_len != ACCELERATION_LEN:
                    print("check head type "+str(TYPE_MSG_BODY_VEL)+" failed;"+" check_LEN:"+str(check_len))
                    continue
                elif head_type == TYPE_BODY_ACCELERATION and check_len != BODY_ACCELERATION_LEN:
                    print("check head type "+str(TYPE_BODY_ACCELERATION)+" failed;"+" check_LEN:"+str(check_len))
                    continue
                elif head_type == TYPE_ACCELERATION and check_len != ACCELERATION_LEN:
                    print("check head type "+str(TYPE_ACCELERATION)+" failed;"+" ckeck_LEN:"+str(check_len))
                    continue
                check_sn = ser.read().hex()
                head_crc8 = ser.read().hex()
                crc16_H_s = ser.read().hex()
                crc16_L_s = ser.read().hex()

                # 读取并解析IMU数据
                if head_type == TYPE_IMU:
                    data_s = ser.read(int(IMU_LEN, 16))
                    IMU_DATA = struct.unpack('12f ii',data_s[0:56])
                    #print(IMU_DATA)
                    # print("Gyroscope_X(rad/s): " + str(IMU_DATA[0]))
                    # print("Gyroscope_Y(rad/s) : " + str(IMU_DATA[1]))
                    # print("Gyroscope_Z(rad/s) : " + str(IMU_DATA[2]))
                    # print("Accelerometer_X(m/s^2) : " + str(IMU_DATA[3]))
                    # print("Accelerometer_Y(m/s^2) : " + str(IMU_DATA[4]))
                    # print("Accelerometer_Z(m/s^2) : " + str(IMU_DATA[5]))
                    # print("Magnetometer_X(mG) : " + str(IMU_DATA[6]))
                    # print("Magnetometer_Y(mG) : " + str(IMU_DATA[7]))
                    # print("Magnetometer_Z(mG) : " + str(IMU_DATA[8]))
                    self._publish_gyro(IMU_DATA[0],IMU_DATA[1],IMU_DATA[2])
                    self._publish_acceleration(IMU_DATA[3],IMU_DATA[4],IMU_DATA[5])
                    self._publish_magnetic(IMU_DATA[6],IMU_DATA[7],IMU_DATA[8])
                    # print("IMU_Temperature : " + str(IMU_DATA[9]))
                    # print("Pressure : " + str(IMU_DATA[10]))
                    # print("Pressure_Temperature : " + str(IMU_DATA[11]))
                    # print("Timestamp(us) : " + str(IMU_DATA[12]))
                # 读取并解析AHRS数据
                elif head_type == TYPE_AHRS:
                    data_s = ser.read(int(AHRS_LEN, 16))
                    AHRS_DATA = struct.unpack('10f ii',data_s[0:48])
                    #print(AHRS_DATA)
                    # print("RollSpeed(rad/s): " + str(AHRS_DATA[0]))
                    # print("PitchSpeed(rad/s) : " + str(AHRS_DATA[1]))
                    # print("HeadingSpeed(rad) : " + str(AHRS_DATA[2]))
                    # print("Roll(rad) : " + str(AHRS_DATA[3]))
                    # print("Pitch(rad) : " + str(AHRS_DATA[4]))
                    # print("Heading(rad) : " + str(AHRS_DATA[5]))
                    Q1W=struct.unpack('f', data_s[24:28])[0]
                    Q2X=struct.unpack('f', data_s[28:32])[0]
                    Q3Y=struct.unpack('f', data_s[32:36])[0]
                    Q4Z=struct.unpack('f', data_s[36:40])[0]
                    Q2EULER=tfs.euler.quat2euler([Q1W,Q2X,Q3Y,Q4Z],"sxyz") #使用前需要安装依赖pip install transforms3d -i https://pypi.tuna.tsinghua.edu.cn/simple
                    # euler=[0,1,2]
                    # euler[0]=Q2EULER[0]*360/2/PI
                    # euler[1]=Q2EULER[1]*360/2/PI
                    # euler[2]=Q2EULER[2]*360/2/PI
                    # print("euler_x: "+str(euler[0]))
                    # print("euler_y: "+str(euler[1]))
                    # print("euler_z: "+str(euler[2]))
                    self._publish_euler(Q2EULER[0],Q2EULER[1],Q2EULER[2])
                    self._publish_quaternion(Q2X,Q3Y,Q4Z,Q1W)
                    self._broadcast_tf(Q2X,Q3Y,Q4Z,Q1W)
                    # print("Q1 : " + str(AHRS_DATA[6]))
                    # print("Q2 : " + str(AHRS_DATA[7]))
                    # print("Q3 : " + str(AHRS_DATA[8]))
                    # print("Q4 : " + str(AHRS_DATA[9]))
                    # print("Timestamp(us) : " + str(AHRS_DATA[10]))
                # 读取并解析INSGPS数据
                elif head_type == TYPE_INSGPS:
                    data_s = ser.read(int(INSGPS_LEN, 16))
                    INSGPS_DATA = struct.unpack('16f ii',data_s[0:72])
                    # print(INSGPS_DATA)
                    # print("BodyVelocity_X:(m/s)" + str(INSGPS_DATA[0]))
                    # print("BodyVelocity_Y:(m/s)" + str(INSGPS_DATA[1]))
                    # print("BodyVelocity_Z:(m/s)" + str(INSGPS_DATA[2]))
                    # print("BodyAcceleration_X:(m/s^2)" + str(INSGPS_DATA[3]))
                    # print("BodyAcceleration_Y:(m/s^2)" + str(INSGPS_DATA[4]))
                    # print("BodyAcceleration_Z:(m/s^2)" + str(INSGPS_DATA[5]))
                    # print("Location_North:(m)" + str(INSGPS_DATA[6]))
                    # print("Location_East:(m)" + str(INSGPS_DATA[7]))
                    # print("Location_Down:(m)" + str(INSGPS_DATA[8]))
                    # print("Velocity_North:(m)" + str(INSGPS_DATA[9]))
                    # print("Velocity_East:(m/s)" + str(INSGPS_DATA[10]))
                    # print("Velocity_Down:(m/s)" + str(INSGPS_DATA[11]))
                    # print("Acceleration_North:(m/s^2)" + str(INSGPS_DATA[12]))
                    # print("Acceleration_East:(m/s^2)" + str(INSGPS_DATA[13]))
                    # print("Acceleration_Down:(m/s^2)" + str(INSGPS_DATA[14]))
                    # print("Pressure_Altitude:(m)" + str(INSGPS_DATA[15]))
                    # print("Timestamp:(us)" + str(INSGPS_DATA[16]))
                # 读取并解析GPS数据
                elif head_type == TYPE_GEODETIC_POS:
                    data_s = ser.read(int(GEODETIC_POS_LEN, 16))
                    # print(" Latitude:(rad)" + str(struct.unpack('d', data_s[0:8])[0]))
                    # print("Longitude:(rad)" + str(struct.unpack('d', data_s[8:16])[0]))
                    # print("Height:(m)" + str(struct.unpack('d', data_s[16:24])[0]))
                elif head_type == TYPE_SYS_STATE:
                    data_s = ser.read(int(SYS_STATE_LEN, 16))
                    # print("Unix_time:" + str(struct.unpack('i', data_s[4:8])[0]))
                    # print("Microseconds:" + str(struct.unpack('i', data_s[8:12])[0]))
                    # print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    # print("System_Z(m/s^2): " + str(struct.unpack('f', data_s[56:60])[0]))
                elif head_type == TYPE_BODY_ACCELERATION:
                    data_s = ser.read(int(BODY_ACCELERATION_LEN, 16))
                    # print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    # print("BodyAcceleration_Z(m/s^2): " + str(struct.unpack('f', data_s[8:12])[0]))
                elif head_type == TYPE_ACCELERATION:
                    data_s = ser.read(int(ACCELERATION_LEN, 16))
                    # print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    # print("Acceleration_Z(m/s^2): " + str(struct.unpack('f', data_s[8:12])[0]))
                elif head_type == TYPE_MSG_BODY_VEL:
                    data_s = ser.read(int(ACCELERATION_LEN, 16))
                    # print(" System_status:" + str(struct.unpack('d', data_s[0:2])[0]))
                    Velocity_X = struct.unpack('f', data_s[0:4])[0]   # 解析第一个双精度浮点数
                    Velocity_Y = struct.unpack('f', data_s[4:8])[0]  # 解析第二个双精度浮点数
                    Velocity_Z = struct.unpack('f', data_s[8:12])[0] # 解析第三个双精度浮点数
                    print(f"Velocity_X: {Velocity_X}, Velocity_Y: {Velocity_Y}, Velocity_Z: {Velocity_Z}")
                    # print("Velocity_X(m/s): " + str(struct.unpack('f', data_s[0:4])[0]))
                    # print("Velocity_Y(m/s): " + str(struct.unpack('f', data_s[4:8])[0]))
                    # print("Velocity_Z(m/s): " + str(struct.unpack('f', data_s[8:12])[0]))

            except Exception as e:
                self.get_logger().warn(f'串口读取异常: {e}')
                time.sleep(0.01)

    # 发布TF
    def _broadcast_tf(self, qx: float, qy: float, qz: float, qw: float):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame_id
        t.child_frame_id = self.child_frame_id
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    # 发布四元数
    def _publish_quaternion(self, qx: float, qy: float, qz: float, qw: float):
        msg = QuaternionStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.child_frame_id
        msg.quaternion.x = qx
        msg.quaternion.y = qy
        msg.quaternion.z = qz
        msg.quaternion.w = qw
        self.q_pub.publish(msg)

    # 发布欧拉角
    def _publish_euler(self, roll: float, pitch: float, yaw: float):
        now = self.get_clock().now().to_msg()
        # radians
        rad_msg = Vector3Stamped()
        rad_msg.header.stamp = now
        rad_msg.header.frame_id = self.child_frame_id
        rad_msg.vector.x = roll
        rad_msg.vector.y = pitch
        rad_msg.vector.z = yaw
        self.euler_rad_pub.publish(rad_msg)
        # degrees
        deg_msg = Vector3Stamped()
        deg_msg.header.stamp = now
        deg_msg.header.frame_id = self.child_frame_id
        deg_msg.vector.x = roll * RAD2DEG
        deg_msg.vector.y = pitch * RAD2DEG
        deg_msg.vector.z = yaw * RAD2DEG
        self.euler_deg_pub.publish(deg_msg)

    def _publish_gyro(self, gx: float, gy: float, gz: float):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.child_frame_id
        msg.vector.x = gx
        msg.vector.y = gy
        msg.vector.z = gz
        self.gyro_pub.publish(msg)

    def _publish_acceleration(self, ax: float, ay: float, az: float):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.child_frame_id
        msg.vector.x = ax
        msg.vector.y = ay
        msg.vector.z = az
        self.acc_pub.publish(msg)

    def _publish_magnetic(self, mx: float, my: float, mz: float):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.child_frame_id
        msg.vector.x = mx
        msg.vector.y = my
        msg.vector.z = mz
        self.mag_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # 仅在未关闭时调用，避免重复 shutdown 报错
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
