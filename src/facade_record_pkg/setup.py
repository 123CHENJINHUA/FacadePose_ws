from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'facade_record_pkg'

setup(
    name=package_name,
    version='0.0.1',  # initial release
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cjh',
    maintainer_email='jchenjb@connect.ust.hk',
    description='ROS2 bag recorder for RealSense and IMU topics',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
