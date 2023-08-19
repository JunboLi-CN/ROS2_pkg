from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='truck_detector_ros2',
            executable='truck_detector',
            name='truck_detection_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'cam_topic': 'camera_image'},
                {'cam_pos_topic': '/mavros/local_position/pose'},
                {'allow_delay': True},
                {'engine': '/workspaces/isaac_ros-dev/src/truck_detector_ros2/model/model.trt'},
                {'device': 'cuda:0'},
                {'scale_factor': 0.015},
                {'update_frequency': 100}
            ]
        )
    ])