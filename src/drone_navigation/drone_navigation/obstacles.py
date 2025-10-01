import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import open3d as o3d
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from std_msgs.msg import Float32MultiArray 

class Obstacles(Node):
    def __init__(self):
        super().__init__('obstacles_node')

        # Subscribe to the drone's odometry to get height
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.odom_callback,
            10
        )

        # Subscribe to the point cloud
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            '/filtered_pointcloud',
            self.point_cloud_callback,
            10
        )

        self.global_publisher = self.create_publisher(PointCloud2, '/map', 10)
        self.circle_publisher = self.create_publisher(Float32MultiArray, '/circle_data', 10)
        self.global_map = o3d.geometry.PointCloud()

        # Initialize matplotlib plot
        plt.ion()
        self.fig, self.ax = plt.subplots()

        # Variable to store current drone height
        self.current_height = None
        self.x = None
        self.y = None

    def odom_callback(self, msg):
        # Extract the height (z-position) from the odometry message
        self.current_height = msg.pose.pose.position.z
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        #self.get_logger().info(f'Drone height: {self.current_height}')

    def point_cloud_callback(self, msg):
        if self.current_height is None:
            self.get_logger().warn('Drone height not available yet')
            return

        # Convert ROS PointCloud2 message to a list of points
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if len(points_list) == 0:
            self.get_logger().warn('Received an empty point cloud')
            return

        # Convert list to numpy array
        cloud_arr = np.array(points_list, dtype=np.float32)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_arr)

        # Random downsampling
        sample_ratio = 0.05  # Keep 10% of the points
        cloud_filtered = cloud.random_down_sample(sample_ratio)

        points = np.asarray(cloud_filtered.points)
        points = points[np.isfinite(points).all(axis=1)] 

        # Convert to ROS PointCloud2 message
        self.global_map.points.extend(points.tolist())
        header = msg.header
        header.frame_id = 'world'
        output_points = np.array(points)
        output_cloud = pc2.create_cloud_xyz32(header, np.asarray(self.global_map.points))
        self.global_publisher.publish(output_cloud)
        points = np.asarray(self.global_map.points)

        # Perform clustering on the 3D (x, y, z) coordinates
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
        labels = clustering.labels_

        # Find number of clusters
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        self.get_logger().info(f'Number of clusters found: {num_clusters}')

        # Clear previous plot
        self.ax.clear()

        # Filter points based on height and fit a circle for each cluster
        all_centers = []
        all_radii = []
        height_threshold = 0.2  # Allow points within ±0.2 meters of the drone's height
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # Skip noise points

            # Get all points in the current cluster
            cluster_points = points[labels == cluster_id]

            # Filter points within the cluster by height
            points_at_height = cluster_points[
                np.abs(cluster_points[:, 2] - self.current_height) < height_threshold
            ]

            if len(points_at_height) == 0:
                self.get_logger().info(f'No points at the drone height in cluster {cluster_id}')
                continue

            self.get_logger().info(f'Number of points at drone height in cluster {cluster_id}: {len(points_at_height)}')

            # Fit circle to the 2D (x, y) points at the drone's height
            circle_points = points_at_height[:, :2]
            if len(circle_points) < 3:
                self.get_logger().warn(f'Not enough points to fit a circle for cluster {cluster_id}')
                continue

            center, radius = self.fit_circle(circle_points)
            self.get_logger().info(f'Fitted circle for cluster {cluster_id}: Center={center}, Radius={radius}')

            # Plot fitted circle
            #self.plot_circle(center, radius)
            all_centers.append(center)
            all_radii.append(radius)
        # self.plot_all_circles(all_centers, all_radii)
        circle_data_msg = Float32MultiArray()
        for center, radius in zip(all_centers, all_radii):
            circle_data_msg.data.extend([center[0], center[1], radius])
        self.circle_publisher.publish(circle_data_msg)

        # Update the plot
        # plt.draw()
        # plt.pause(0.001)

    def fit_circle(self, points):
        # Circle fitting using least squares
        def calc_radius(x):
            return np.sqrt((points[:, 0] - x[0])**2 + (points[:, 1] - x[1])**2)

        def loss_function(x):
            return np.sum((calc_radius(x) - np.mean(calc_radius(x)))**2)

        x0 = np.mean(points, axis=0)  # Initial guess: mean of the points
        res = least_squares(loss_function, x0)
        center = res.x
        radius = np.mean(calc_radius(center))

        return center, radius
    def publish_circle_data(self, centers, radii):
        """ Publish circle centers and radii as Float32MultiArray """
        if len(centers) == 0:
            self.get_logger().info('No circles to publish.')
            return

        circle_data = Float32MultiArray()
        data = []

        # Flatten the centers and radii into a single list
        for center, radius in zip(centers, radii):
            data.extend(center.tolist())  # Add center (x, y)
            data.append(radius)           # Add radius

        circle_data.data = data
        self.circle_publisher.publish(circle_data)


    def plot_all_circles(self, centers, radii):
        """ Plot all circles given their centers and radii """
        if len(centers) == 0:
            self.get_logger().info('No circles to plot.')
            return

        # Plot each circle and adjust the axis limits to fit all circles
        all_x = []
        all_y = []
        for center, radius in zip(centers, radii):
            circle = plt.Circle(center, radius, color='b', fill=False)
            self.ax.add_patch(circle)
            all_x.append(center[0])
            all_y.append(center[1])
        if self.x is not None and self.y is not None:
            self.ax.plot(self.x, self.y, 'ro', label='Drone Position', markersize=5)

        # Set axis limits dynamically based on all circle centers and radii
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        margin = 1  # Extra margin around the circles
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim([min_x - margin, max_x + margin])
        self.ax.set_ylim([min_y - margin, max_y + margin])

def main(args=None):
    rclpy.init(args=args)
    obstacle_node = Obstacles()
    rclpy.spin(obstacle_node)
    obstacle_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

