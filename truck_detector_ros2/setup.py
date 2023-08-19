import os
from glob import glob
from setuptools import setup

package_name = 'truck_detector_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'model'), glob('model/*.trt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admin',
    maintainer_email='admin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'truck_detector = truck_detector_ros2.truck_detector:main'
        ],
    },
)
