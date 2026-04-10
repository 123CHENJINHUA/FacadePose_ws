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
            
        # 过滤：移除深度大于3.0米的点（假定 depth 在 z 分量）
        try:
            if point_cloud is None:
                return []
            # 仅在点云有至少3个分量时按 z 分量过滤
            if isinstance(point_cloud, np.ndarray) and point_cloud.shape[1] >= 3:
                depth = point_cloud[:, 2]
                mask = np.isfinite(depth) & (depth <= 3.0)
                point_cloud = point_cloud[mask]
        except Exception:
            # 若出现任何问题，退回到原始点云（不过不抛出异常）
            pass

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
    
    def select_best_plane(self, planes, return_debug: bool = False):
        """选择最佳平面，并可选返回调试信息。

        旧逻辑：先按法向量方向分组挑 best group，再在组内按点数(面积)挑最大。
        新逻辑：对每个 plane 计算综合分数，同时考虑：
          - 方向相似度（与 camera_vector 对齐程度，带软阈值）
          - 平面规模（用点数的 log 归一化）

        Args:
            planes: list of plane dict
            return_debug: when True returns (best_plane, debug)

        Returns:
            best_plane if return_debug is False
            (best_plane, debug) if return_debug is True
        """
        debug = {
            'num_planes': 0,
            # keep previous fields for compatibility / visualization
            'groups': [],  # list of groups, each group: list of plane indices
            'group_avg_similarity': [],
            'group_best_plane_index': [],
            'best_group_index': None,
            'best_plane_index': None,
            'camera_vector': np.array([0.0, 0.0, -1.0], dtype=np.float64),
            'normal_angle_thresh_deg': 10.0,
            # new scoring fields
            'score_mode': 'combined_dir_area',
            'dir_weight': 0.6,
            'area_weight': 0.4,
            'per_plane': [],  # list of dicts per plane: {idx, npts, sim, dir_score, area_score, score}
            'max_npts': 0,
        }

        def _ret(best_plane_, best_idx_):
            debug['best_plane_index'] = best_idx_
            if return_debug:
                return best_plane_, debug
            return best_plane_

        if not planes:
            debug['num_planes'] = 0
            return _ret(None, None)

        debug['num_planes'] = len(planes)

        # ---------- (A) group planes by similar normals (kept for debug/visualization) ----------
        groups: list[list[int]] = []
        representatives: list[np.ndarray] = []
        cos_th = float(np.cos(np.deg2rad(debug['normal_angle_thresh_deg'])))

        for i, plane in enumerate(planes):
            n = np.array(plane.get('normal', [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-9:
                groups.append([i])
                representatives.append(n)
                continue

            n = n / max(np.linalg.norm(n), 1e-12)
            is_grouped = False
            for gi, repn in enumerate(representatives):
                if np.all(np.isfinite(repn)) and np.linalg.norm(repn) > 1e-9:
                    repn_u = repn / max(np.linalg.norm(repn), 1e-12)
                    if float(np.dot(n, repn_u)) > cos_th:
                        groups[gi].append(i)
                        is_grouped = True
                        break
            if not is_grouped:
                groups.append([i])
                representatives.append(n)

        debug['groups'] = [g[:] for g in groups]

        # group avg similarity (same as before)
        camera_vector = np.array(debug['camera_vector'], dtype=np.float64).reshape(3)
        if np.linalg.norm(camera_vector) > 1e-9:
            camera_vector = camera_vector / np.linalg.norm(camera_vector)

        max_avg_similarity = -1e9
        best_group_idx = None
        for gi, group in enumerate(groups):
            sims = []
            for idx in group:
                n = np.array(planes[idx].get('normal', [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
                if not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-9:
                    continue
                n = n / max(np.linalg.norm(n), 1e-12)
                sims.append(float(np.dot(n, camera_vector)))
            avg_similarity = float(np.mean(sims)) if len(sims) else float('nan')
            debug['group_avg_similarity'].append(avg_similarity)
            if np.isfinite(avg_similarity) and avg_similarity > max_avg_similarity:
                max_avg_similarity = avg_similarity
                best_group_idx = gi
        debug['best_group_index'] = best_group_idx

        # group best plane index by size (same as before; best effort)
        try:
            debug['group_best_plane_index'] = [
                (max(g, key=lambda j: (len(planes[j].get('pcd', None).points) if hasattr(planes[j].get('pcd', None), 'points') else 0)))
                if g else None
                for g in groups
            ]
        except Exception:
            debug['group_best_plane_index'] = [None for _ in groups]

        # ---------- (B) combined scoring per-plane ----------
        npts_list: list[int] = []
        for p in planes:
            pcd = p.get('pcd', None)
            try:
                npts = int(len(pcd.points))
            except Exception:
                # fallback: if some caller passes ndarray points
                try:
                    arr = np.asarray(pcd)
                    npts = int(arr.shape[0])
                except Exception:
                    npts = 0
            npts_list.append(max(npts, 0))

        max_npts = int(max(npts_list)) if npts_list else 0
        debug['max_npts'] = max_npts

        # direction soft-threshold
        cos0 = float(np.cos(np.deg2rad(debug['normal_angle_thresh_deg'])))

        best_idx = None
        best_score = -1e18
        for i, plane in enumerate(planes):
            n = np.array(plane.get('normal', [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
            n_norm = float(np.linalg.norm(n))
            if not np.all(np.isfinite(n)) or n_norm < 1e-9:
                sim = float('nan')
                dir_score = 0.0
            else:
                n_u = n / n_norm
                sim = float(np.clip(np.dot(n_u, camera_vector), -1.0, 1.0))
                # soft threshold to [0, 1]
                dir_score = float(np.clip((sim - cos0) / max(1.0 - cos0, 1e-9), 0.0, 1.0))

            npts = npts_list[i]
            if max_npts <= 0:
                area_score = 0.0
            else:
                # log-normalized size in [0, 1]
                area_score = float(np.log1p(npts) / max(np.log1p(max_npts), 1e-9))

            score = float(debug['dir_weight']) * dir_score + float(debug['area_weight']) * area_score

            debug['per_plane'].append({
                'idx': i,
                'npts': int(npts),
                'sim': float(sim) if np.isfinite(sim) else float('nan'),
                'dir_score': float(dir_score),
                'area_score': float(area_score),
                'score': float(score),
            })

            if score > best_score:
                best_score = score
                best_idx = i

        best_plane = planes[best_idx] if best_idx is not None else None
        return _ret(best_plane, best_idx)


