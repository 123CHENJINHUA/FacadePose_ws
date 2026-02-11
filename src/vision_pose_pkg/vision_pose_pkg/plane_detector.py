#!/usr/bin/env python3
"""
平面检测模块
参考normal_imu2.py的平面检测方法
使用Open3D进行平面分割
"""

import os
import numpy as np
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    # 尝试使用GPU
    try:
        device = o3d.core.Device("CUDA:0")
        print("Using GPU for Open3D operations")
        # device = o3d.core.Device("CPU:0")
        # print("GPU not available, using CPU for Open3D operations")
    except:
        device = o3d.core.Device("CPU:0")
        print("GPU not available, using CPU for Open3D operations")
except ImportError:
    OPEN3D_AVAILABLE = False
    device = None
    print("Open3D not available, plane detection will be limited")


class PlaneDetector:
    """平面检测器"""
    
    def __init__(self, min_points=300, distance_threshold=0.02, voxel_size=0.02):
        self.min_points = min_points
        self.distance_threshold = distance_threshold
        self.voxel_size = voxel_size
        self.gpu_available = OPEN3D_AVAILABLE and device is not None and device.get_type() == o3d.core.Device.DeviceType.CUDA
        
    def find_planes(self, point_cloud):
        """
        在点云中查找多个平面
        参数:
            point_cloud: Nx3 numpy array
        返回:
            planes: list of dict with 'normal', 'center', 'pcd'
        """
        if not OPEN3D_AVAILABLE:
            print("Open3D not available")
            return []
            
        if point_cloud is None or len(point_cloud) < self.min_points:
            return []

        planes = []
        
        # 尝试使用GPU加速
        if self.gpu_available:
            try:
                planes = self._find_planes_gpu(point_cloud)
            except Exception as e:
                print(f"GPU processing failed, falling back to CPU: {e}")
                planes = self._find_planes_cpu(point_cloud)
        else:
            planes = self._find_planes_cpu(point_cloud)
            
        return planes
    
    def _find_planes_gpu(self, point_cloud):
        """使用GPU加速的平面检测"""
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor(point_cloud, dtype=o3d.core.float32, device=device)
        
        # 降采样
        downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        planes = []
        rest = downpcd
        
        while len(rest.point.positions) > self.min_points:
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < self.min_points:
                break

            # 复制到CPU
            [a, b, c, d] = plane_model.cpu().numpy()
            normal = np.array([a, b, c])
            
            # 确保法向量指向相机
            if normal[2] > 0:
                normal = -normal
                d = -d

            plane_pcd = rest.select_by_index(inliers)
            center = plane_pcd.get_center().cpu().numpy()
            
            # 转换为legacy格式
            legacy_pcd = o3d.geometry.PointCloud()
            legacy_pcd.points = o3d.utility.Vector3dVector(plane_pcd.point.positions.cpu().numpy())
            
            planes.append({
                'normal': normal,
                'center': center,
                'pcd': legacy_pcd,
                'd': d
            })
            
            rest = rest.select_by_index(inliers, invert=True)
        
        return planes
    
    def _find_planes_cpu(self, point_cloud):
        """使用CPU的平面检测"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # 降采样
        downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        planes = []
        rest = downpcd
        
        while len(rest.points) > self.min_points:
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < self.min_points:
                break

            [a, b, c, d] = plane_model
            normal = np.array([a, b, c])
            
            # 确保法向量指向相机
            if normal[2] > 0:
                normal = -normal
                d = -d

            plane_pcd = rest.select_by_index(inliers)
            center = plane_pcd.get_center()
            
            planes.append({
                'normal': normal,
                'center': center,
                'pcd': plane_pcd,
                'd': d
            })
            
            rest = rest.select_by_index(inliers, invert=True)
        
        return planes
    
    def select_best_plane(self, planes):
        """
        选择最佳平面（最大的平面）
        """
        if not planes:
            return None
        
        best_plane = None

        # 1. Group planes with similar normals (angle < 10 degrees)
        groups = []
        for plane in planes:
            is_grouped = False
            for group in groups:
                # Compare with the normal of the first plane in the group
                representative_normal = group[0]['normal']
                if np.dot(plane['normal'], representative_normal) > np.cos(np.deg2rad(10)):
                    group.append(plane)
                    is_grouped = True
                    break
            if not is_grouped:
                groups.append([plane])

        # 2. Calculate average similarity for each group and find the best group
        best_group = None
        max_avg_similarity = -1
        camera_vector = np.array([0, 0, -1])
        if groups:
            for group in groups:
                similarities = [np.dot(p['normal'], camera_vector) for p in group]
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > max_avg_similarity:
                    max_avg_similarity = avg_similarity
                    best_group = group

        # 3. From the best group, select the plane with the most points
        if best_group:
            best_plane = max(best_group, key=lambda p: len(p['pcd'].points))
                
        return best_plane
    
    
