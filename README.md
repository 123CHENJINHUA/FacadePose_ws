1. 把numpy版本降级
2. 在命令行使用 export PYTHONPATH=$PYTHONPATH:/home/cjh/anaconda3/envs/facade_pose_fusion/lib/python3.10/site-packages

export PYTHONPATH=$PYTHONPATH:/home/cjh/anaconda3/envs/facade_pose/lib/python3.10/site-packages

ros2 launch launch_pkg_cpp start.launch.py bag_name:=facade_selected_20251124_184438