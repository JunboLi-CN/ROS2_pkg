from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_ros2',
            executable='camera',
            name='camera_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'width': '1280'},
                {'height': '720'},
                {'fps': '60'}
            ]
        )
    ])