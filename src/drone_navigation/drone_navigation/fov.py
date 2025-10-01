import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations
from ament_index_python.packages import get_package_share_directory
import os


class PointCloudFilterNode(Node):
    def __init__(self):
        super().__init__('pointcloud_filter_node')

        # Declare parameters
        self.declare_parameter('fov_angle', 2.09)  # Field of view angle in radians
        self.declare_parameter('max_distance', 4.0)  # Max distance in meters
        package_share = get_package_share_directory('drone_navigation')
        default_pcd = os.path.join(package_share, 'maps', 'medium.pcd')

        self.declare_parameter('pcd_file_path', default_pcd)

        # Initialize variables
        self.drone_position = None
        self.drone_orientation = None

        # Subscribers
        self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.odometry_callback,
            10
        )
        # Publisher
        self.filtered_publisher = self.create_publisher(
            PointCloud2,
            'filtered_pointcloud',
            10
        )

        # Process the PCD file
        self.process_pcd_file()

    def odometry_callback(self, msg):
        # Update drone position and orientation
        self.drone_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.drone_orientation = msg.pose.pose.orientation

        # Convert quaternion to yaw
        _, _, self.drone_yaw = tf_transformations.euler_from_quaternion([
            self.drone_orientation.x,
            self.drone_orientation.y,
            self.drone_orientation.z,
            self.drone_orientation.w
        ])

        # Process and filter the point cloud once the odometry data is received
        self.process_pcd_file()

    def process_pcd_file(self):
        if self.drone_position is None or self.drone_orientation is None:
            self.get_logger().info('Waiting for odometry data...')
            return

        # Read the PCD file using Open3D
        pcd_file_path = self.get_parameter('pcd_file_path').get_parameter_value().string_value
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        sample_ratio = 0.05  # Keep 5% of the points
        cloud_filtered = pcd.random_down_sample(sample_ratio)
        # Convert Open3D point cloud to numpy array
        points = np.asarray(cloud_filtered.points)
        
 
        # Apply filtering based on the drone's position and orientation
        filtered_points = self.filter_points(points)

        # Convert filtered points to PointCloud2 msg
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'world'  # You can set this to the appropriate frame ID
        pc2_msg = pc2.create_cloud_xyz32(header, filtered_points)

        # Publish the filtered point cloud
        self.filtered_publisher.publish(pc2_msg)

    def filter_points(self, points):
        fov_angle = self.get_parameter('fov_angle').get_parameter_value().double_value
        max_distance = self.get_parameter('max_distance').get_parameter_value().double_value

        # Translate points to the drone's coordinate system
        translated_points = points - self.drone_position

        # Convert points to polar coordinates
        distances = np.linalg.norm(translated_points, axis=1)
        angles = np.arctan2(translated_points[:, 1], translated_points[:, 0]) - self.drone_yaw
        
        # Normalize angles
        angles = np.arctan2(np.sin(angles), np.cos(angles))  # Wrap to [-pi, pi]

        # Filter points within the FoV cone
        in_fov = (distances <= max_distance) & (np.abs(angles) <= fov_angle / 2.0)
        return points[in_fov]

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

