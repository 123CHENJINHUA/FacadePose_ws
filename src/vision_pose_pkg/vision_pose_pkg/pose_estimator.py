#!/usr/bin/env python3
"""
视觉姿态估计模块
基于平面检测结果计算相机姿态
参考normal_imu2.py的方法
"""

import numpy as np
import cv2
import rclpy


class VisionPoseEstimator:
    """视觉姿态估计器"""
    
    def __init__(self, camera_matrix, dist_coeffs):
        """
        参数:
            camera_matrix: 3x3相机内参矩阵
            dist_coeffs: 畸变系数
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def normalize(self, v):
        """归一化向量"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    
    def align_vectors(self, v1, v2):
        """Computes the rotation matrix that aligns v1 to v2."""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        if np.linalg.norm(axis) == 0:
            # If vectors are collinear
            if np.allclose(v1, v2):
                return np.eye(3)
            else:
                # If they are opposite, rotate 180 degrees around an arbitrary axis (e.g., X-axis)
                rx = np.pi
                cos_rx, sin_rx = np.cos(rx), np.sin(rx)

                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, cos_rx, -sin_rx],
                    [0, sin_rx, cos_rx]
                ])
                return rotation_matrix
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def calculate_rotation_angle_from_line(self, plane, image, gravity_vector=None):
        """
        Calculates the rotation angle around the normal by finding the longest line
        in the plane's 2D projection and determining if it's horizontal or vertical
        based on gravity vector.
        """
        # 1. Get 2D points and find bounding box
        plane_points_3d = np.asarray(plane['pcd'].points)
        if plane_points_3d.shape[0] < 20: # Need enough points to form a reliable plane
            # print("Not enough 3D points to form a plane")
            return None, None, None

        plane_points_2d, _ = cv2.projectPoints(plane_points_3d, np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs)
        plane_points_2d = np.squeeze(plane_points_2d).astype(int)

        if plane_points_2d.ndim == 1 or len(plane_points_2d) < 2:
            # print("Invalid 2D points")
            return None, None, None

        x, y, w, h = cv2.boundingRect(plane_points_2d)
        
        # Ensure the bounding box is of a minimum size
        if w < 20 or h < 20:
            # print("Bounding box is too small")
            return None, None, None

        # 2. Crop the image, add some padding
        padding = 10
        x_start, y_start = max(x - padding, 0), max(y - padding, 0)
        x_end, y_end = min(x + w + padding, image.shape[1]), min(y + h + padding, image.shape[0])
        
        cropped_image = image[y_start:y_end, x_start:x_end]
        if cropped_image.size == 0:
            # print("Cropped image is empty")
            return None, None, None

        # 3. Grayscale and Canny
        gray_crop = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_crop, 50, 150)

        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)

        if lines is None:
            # print("No lines found")
            return None, None, None

        # 5. Find the longest line
        longest_line = None
        max_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > max_len:
                max_len = length
                longest_line = (x1, y1, x2, y2)

        if longest_line is None:
            # print("Failed to find longest line")
            return None, None, None
        
        # Ensure normal is a numpy array and normalized
        normal = np.array(plane['normal'], dtype=np.float64).reshape(3)
        norm_val = np.linalg.norm(normal)
        if norm_val == 0 or np.isnan(norm_val):
            # print("Invalid normal vector")
            return None, None, None
        normal = normal / norm_val

        # Ensure normal points toward the camera (z should be negative in camera frame)
        if normal[2] > 0:
            normal = -normal

        x1, y1, x2, y2 = longest_line

        # 计算图像直线的齐次表示 (l^T * x = 0)
        p1_h = np.array([x1, y1, 1.0])
        p2_h = np.array([x2, y2, 1.0])
        l = np.cross(p1_h, p2_h)
        l = self.normalize(l)

        # 计算反投影平面
        n_line = self.camera_matrix.T @ l  # 反投影得到过相机中心与图像直线的平面法向量
        n_line = self.normalize(n_line)

        # 计算直线的三维方向（位于平面内）
        v_c = np.cross(normal, n_line)
        v_c_norm = np.linalg.norm(v_c)
        if v_c_norm < 1e-6:
            print("Degenerate line direction; fallback to normal alignment only")
            v_c = None
        else:
            v_c = v_c / v_c_norm

        if gravity_vector is not None:
            g = self.normalize(gravity_vector)
        else:
            g = None

        # 第一步：对齐法向量，使平面法向量与相机参考法向量 [0,0,-1] 一致
        R_normal = self.align_vectors(np.array([0.0, 0.0, -1.0]), normal)

        # 选择平面内基准轴（受重力与直线方向关系影响）
        parallel_thresh = 0.95
        perp_thresh = 0.2
        baseline_axis = np.array([-1.0, 0.0, 0.0])  # 默认 X
        if g is not None and v_c is not None:
            dot_g = abs(np.dot(v_c, g))
            if dot_g > parallel_thresh:
                # 直线方向与重力平行 → 使用相机竖直轴 Y
                baseline_axis = np.array([0.0, 1.0, 0.0])
            elif dot_g < perp_thresh:
                # 直线方向与重力垂直 → 使用相机水平轴 X
                baseline_axis = np.array([1.0, 0.0, 0.0])
            else:
                return None, None, None

        '''
        Rodrigues版本: R_plane = exp([n]_x θ) 只绕法向量 normal 旋转角度 θ，
        保证已对齐的法向量不被破坏，是“受约束旋转”。
        align_vectors(a_proj, v_c): 计算使 a_proj → v_c 的最小旋转，旋转轴为 a_projxv_c。
        理想情况下该轴与 normal 重合，但受数值误差或之前对 v_c 翻转的处理影响，轴可能略偏离 normal,造成法向量被再次扰动。
        '''
        # 将基准轴变换到已对齐法向量后的坐标系
        a = R_normal @ baseline_axis

        # 投影到平面内
        a_proj = a - np.dot(a, normal) * normal
        a_proj_norm = np.linalg.norm(a_proj)
        if a_proj_norm < 1e-6 or v_c is None:
            # 无法可靠计算平面内角度，直接返回法向量对齐结果
            R_total = R_normal
        else:
            a_proj = a_proj / a_proj_norm

            # 计算与直线方向的夹角（用于判断是否接近 180°）
            dot_av = np.clip(np.dot(a_proj, v_c), -1.0, 1.0)
            angle_av = np.degrees(np.arccos(dot_av))  # 0~180

            # 仅当接近 180°（例如 >165°）才翻转，避免不必要翻转
            if angle_av > 165.0:
                v_c = -v_c

            # 计算在平面内从 a_proj 旋转到 v_c 的角度与方向
            cross_ap_vc = np.cross(a_proj, v_c)
            numer = np.dot(normal, cross_ap_vc)        # 有符号 sinθ
            denom = np.clip(np.dot(a_proj, v_c), -1.0, 1.0)  # cosθ
            theta = np.arctan2(numer, denom)
            # print("Rotation angle (degrees):", np.degrees(theta))
            # Rodrigues 绕法向量旋转
            rvec_plane = normal * theta
            R_plane, _ = cv2.Rodrigues(rvec_plane)
            R_total = R_plane @ R_normal

        # Convert rotation matrix to rotation vector (rvec)
        rvec, _ = cv2.Rodrigues(R_total)
        rvec = rvec.reshape(3, 1).astype(np.float32)

        # 平移向量为平面中心 (float32)
        center = np.array(plane['center'], dtype=np.float64).reshape(3)
        tvec = center.reshape(3, 1).astype(np.float32)

        # Return the line coordinates relative to the original image for drawing
        viz_line = (x1 + x_start, y1 + y_start, x2 + x_start, y2 + y_start)

        return rvec, tvec, viz_line