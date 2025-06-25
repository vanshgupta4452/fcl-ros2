from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    xacro_file = PathJoinSubstitution([
        FindPackageShare("mr_robot_description"),
        "urdf",
        "mr_robot.xacro"
    ])

    robot_description_content = Command(["xacro ", xacro_file])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description_content
        }]
    )

    return LaunchDescription([robot_state_publisher])
