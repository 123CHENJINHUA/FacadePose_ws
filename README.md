1. 把numpy版本降级
2. 在命令行使用 export PYTHONPATH=$PYTHONPATH:/home/cjh/anaconda3/envs/facade_pose_fusion/lib/python3.10/site-packages

export PYTHONPATH=$PYTHONPATH:/home/cjh/anaconda3/envs/facade_pose/lib/python3.10/site-packages

ros2 launch launch_pkg_cpp start.launch.py bag_name:=facade_selected_20251124_184438



补充文件calibration_result.yaml,和pose_translate_node.py，要求订阅话题fusion_pose，发布话题depth_to_pose/base1 和 depth_to_pose/base2. 这两个发布的话题和fusion_pose是一样的格式，只是做了坐标转化，转换矩阵需要读取calibration_result.yaml,其中yaml里面的内容是深度相机对base1和base2的转换矩阵，分别为RT_depth_cam_to_base1_mean.txt和RT_depth_cam_to_base2_mean.txt