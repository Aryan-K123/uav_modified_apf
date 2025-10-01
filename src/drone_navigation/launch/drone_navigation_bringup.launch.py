from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to bringup launch file
    sjtu_bringup_pkg = get_package_share_directory('sjtu_drone_bringup')
    sjtu_bringup_launch = os.path.join(sjtu_bringup_pkg, 'launch', 'sjtu_drone_bringup.launch.py')

    return LaunchDescription([
        # Include sjtu-drone_bringup launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sjtu_bringup_launch)
        ),

        # fov_node from drone_navigation
        Node(
            package='drone_navigation',
            executable='fov_node',
            name='fov_node',
            output='screen'
        ),

        # obstacle_node from drone_navigation
        Node(
            package='drone_navigation',
            executable='obstacles_node',
            name='obstacles_node',
            output='screen'
        ),

        Node(
            package='drone_navigation',
            executable='rf_node',
            name='rf_node',
            output='screen'
        ),
        Node(
            package='drone_navigation',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        Node(
            package='drone_navigation',
            executable='height_node',
            name='height_node',
            output='screen'
        ),
    ])
