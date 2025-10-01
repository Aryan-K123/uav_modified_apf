import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Twist,Vector3
import math
import random
from geometry_msgs.msg import Point
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge 
import cv2 
import imutils
from sensor_msgs.msg import Image,CameraInfo
import time

def save_failed_run():
    save_path = "run_results.txt"
    try:
        with open(save_path, "r") as f:
            lines = f.readlines()
        run_number = len(lines) + 1
    except FileNotFoundError:
        run_number = 1

    with open(save_path, "a") as f:
        f.write(f"Run {run_number}: didn't work\n")
class APF(Node):
    def __init__(self):
        super().__init__('apf_node')

        # Create a subscription to the circle data
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/circle_data',
            self.obstacles,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.odom_callback,
            10
        )
        self.velocity_publisher = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.publisher_ = self.create_publisher(Point, 'point_topic', 10)
        self.subscription = self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.listener_callback, 10)
        self.direction_subscriber = self.create_subscription(Vector3,'/rf_direction', self.source_direction,10)

        # Parameters
        self.br = CvBridge()
        self.red_detected = False
        self.drone_radius = 0.2
        self.drone_path = []
        self.k_att = 2.0
        self.k_rep = 2.0
        self.d0 = 2.0
        self.z = None
        self.x = None
        self.y = None
        self.goal = np.array([3, 2])
        self.temp_goal = None  # Initialize as the main goal
        self.grid_resolution = 0.1
        self.start = None
        self.current_position = None
        self.trajectory = None
        self.current_index = 0
        self.obstacles = []
        self.trajectory_active = False
        self.temp_goal_updated = False 
        self.initial_trajectory = None  # Initialize initial trajectory
        self.trajectories = []           # Initialize list to store all generated trajectories
        self.optimal_trajectory = []     # Initialize optimal trajectory
        self.closest_trajectory = None
        self.sampling_count = 10
        self.max_steps = 15
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.yaw = 0.0  # Current yaw angle
        self.prev_heading_error = 0.0  # Previous heading error for yaw PID
        self.integral_heading_error = 0.0 
        self.rf_direction = None
        self.k_att_history = []  # To store k_att values over time
        self.k_rep_history = []  # To store k_rep values over time
        self.d0_history = []     # To store d0 values over time
        self.time_steps = []     # To store time steps (e.g., iteration numbers)
        self.computation_times = []
        self.all_minimum_distance_to_obstacle = []
        
        self.simulation_ended = False
        self.file_saved = False
        
        self.fig = plt.figure(figsize=(14, 7))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        plt.ion()
        self.fig2 = plt.figure(figsize = (14,7))
        self.ax11 = self.fig2.add_subplot(121)
        self.ax22 = self.fig2.add_subplot(122)
        plt.ion()
        # To ensure "Reached target point" is printed only once
        self.reached_points = set()

    def source_direction(self,msg):
        self.rf_direction = [msg.x,msg.y]
    def odom_callback(self, msg):
        self.z = msg.pose.pose.position.z
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.yaw = euler_from_quaternion(quaternion)
        if self.start is None:
            self.start = np.array([self.x, self.y])
            self.get_logger().info(f'Initial position: {self.x}, {self.y}')
            if self.temp_goal is None:
                direction = self.goal - self.start
                distance = np.linalg.norm(direction)
                direction_normalized = direction / distance
                #self.temp_goal = self.start + direction_normalized * 2
                if self.rf_direction is not None:
                   self.temp_goal = self.start + self.rf_direction * 2  # Initial temp goal 1m from start
                else:
                    self.temp_goal = self.start + direction_normalized * 2
                    self.get_logger().info(f'Initial temporary goal: {self.temp_goal}')
        
        self.current_position = np.array([self.x, self.y])

        # Only compute trajectory if obstacles are available
        if self.obstacles:
            self.check_and_compute_trajectory()
        else:
            self.get_logger().info("Waiting for obstacle data to compute trajectory.")

    def obstacles(self, msg):
        """Update obstacles list from incoming message"""
        self.obstacles.clear()
        if len(msg.data) % 3 != 0:
            return
        for i in range(0, len(msg.data), 3):
            if i + 2 < len(msg.data):
                center_x = msg.data[i]
                center_y = msg.data[i + 1]
                radius = msg.data[i + 2]
                self.obstacles.append([center_x, center_y, radius])
        self.get_logger().info(f'Updated {len(self.obstacles)} obstacles.')
        if self.simulation_ended == False:
            self.stopping()

    def update_temp_goal(self):
        """
        Update the temporary goal once the drone reaches halfway through the trajectory.
        The drone will deviate from the direction if an obstacle is detected. This method
        checks for obstacles in a loop with 10-degree deviations until a clear path is found.
        """
        direction = self.goal - self.current_position
        distance_to_goal = np.linalg.norm(direction)

        # Calculate the temporary goal along the direct path towards the goal
        temp_goal_candidate = self.current_position + 2*direction / distance_to_goal

        # Check if there is an obstacle in the path
        if self.is_obstacle_in_path(self.current_position, temp_goal_candidate):
            # If an obstacle is in the path, we will attempt deviations of 10 degrees

            # Deviation angle step
            deviation_angle = 10 * math.pi / 180  # 10 degrees in radians

            # Initialize the deviation checks
            deviation_attempts = 0
            max_attempts = 100  # Set a limit for the number of iterations to avoid infinite loop

            # Loop to check for obstacles with increasing deviation on both sides
            while deviation_attempts < max_attempts:
                # Perform deviations at +10 degrees and -10 degrees
                # Left deviation (negative angle)
                rotation_matrix_left = np.array([[math.cos(deviation_angle), -math.sin(deviation_angle)],
                                                [math.sin(deviation_angle), math.cos(deviation_angle)]])
                new_direction_left = np.dot(rotation_matrix_left, direction)
                temp_goal_left = self.current_position + new_direction_left / np.linalg.norm(new_direction_left)

                # Right deviation (positive angle)
                rotation_matrix_right = np.array([[math.cos(-deviation_angle), -math.sin(-deviation_angle)],
                                                [math.sin(-deviation_angle), math.cos(-deviation_angle)]])
                new_direction_right = np.dot(rotation_matrix_right, direction)
                temp_goal_right = self.current_position + new_direction_right / np.linalg.norm(new_direction_right)

                # Check if there is an obstacle on the left deviation path
                obstacle_left = self.is_obstacle_in_path(self.current_position, temp_goal_left)
                # Check if there is an obstacle on the right deviation path
                obstacle_right = self.is_obstacle_in_path(self.current_position, temp_goal_right)

                # If there's no obstacle on the left path, select left as the new path
                if not obstacle_left:
                    temp_goal_candidate = temp_goal_left
                    self.get_logger().info(f"Selected left side with 10-degree deviation: {temp_goal_candidate}")
                    break
                # If there's no obstacle on the right path, select right as the new path
                elif not obstacle_right:
                    temp_goal_candidate = temp_goal_right
                    self.get_logger().info(f"Selected right side with 10-degree deviation: {temp_goal_candidate}")
                    break
                else:
                    # If both sides are blocked, increase the deviation and try again
                    deviation_angle += 10 * math.pi / 180  # Increase the deviation by another 10 degrees
                    deviation_attempts += 1
                    self.get_logger().info(f"Both sides blocked, increasing deviation to {deviation_angle * 180 / math.pi} degrees.")
            else:
                # If the loop ends, meaning no valid path is found
                self.get_logger().warn("Unable to find a valid path after multiple deviations.")

        # Update the temporary goal if it's valid (i.e., no obstacle in the path)
        if np.linalg.norm(direction)<=2:
            self.temp_goal = self.goal
        else:
            self.temp_goal = temp_goal_candidate
            self.get_logger().info(f"Temporary goal updated to: {self.temp_goal}")
            self.temp_goal_updated = True  # Mark temp goal as updated


    def attractive_potential(self, current_position, goal, k_att):
        return 0.5 * k_att * np.linalg.norm(goal - current_position) ** 2

    def repulsive_potential(self, current_position, obstacles, d0, k_rep):
        U_rep = 0
        for obs in obstacles:
            center = np.array(obs[:2])
            radius = obs[2]
            dist_to_center = np.linalg.norm(current_position - center)
            dist_to_boundary = dist_to_center - radius-self.drone_radius
            if dist_to_boundary < d0:
                if dist_to_boundary < 0:
                    dist_to_boundary = 1e-6
                U_rep += -k_rep * np.log(dist_to_boundary / d0)
                #U_rep += 0.000005*k_rep*((1/dist_to_boundary)-1/d0)
        return U_rep

    def total_potential(self, current_position, goal, obstacles, d0, k_att, k_rep):
        U_att = self.attractive_potential(current_position, goal, k_att)
        U_rep = self.repulsive_potential(current_position, obstacles, d0, k_rep)
        return U_att + U_rep

    def compute_gradient(self, q, goal, obstacles, d0, k_att, k_rep, h=1e-5):
        q_x_forward = np.array([q[0] + h, q[1]])
        q_x_backward = np.array([q[0] - h, q[1]])
        grad_U_x = (self.total_potential(q_x_forward, goal, obstacles, d0, k_att, k_rep) -
                    self.total_potential(q_x_backward, goal, obstacles, d0, k_att, k_rep)) / (2 * h)

        q_y_forward = np.array([q[0], q[1] + h])
        q_y_backward = np.array([q[0], q[1] - h])
        grad_U_y = (self.total_potential(q_y_forward, goal, obstacles, d0, k_att, k_rep) -
                    self.total_potential(q_y_backward, goal, obstacles, d0, k_att, k_rep)) / (2 * h)

        return np.array([grad_U_x, grad_U_y])

    def gradient_descent_trajectory_Ut(self, obstacles, d0, k_att, k_rep, alpha=0.1, tolerance=1e-3):
        self.get_logger().info(f"Starting trajectory computation...")
        q = np.array(self.current_position, dtype=float)
        trajectory = np.zeros((self.max_steps, 2))
        trajectory[0] = q
        goal_reached = False

        for step in range(self.max_steps - 1):
            grad_U = self.compute_gradient(trajectory[step], self.temp_goal, obstacles, d0, k_att, k_rep)
            trajectory[step + 1] = trajectory[step] - alpha * grad_U

            if np.linalg.norm(trajectory[step + 1] - self.goal) < tolerance:
                self.get_logger().info(f"Goal reached in {step + 1} steps.")
                goal_reached = True
                break

        if not goal_reached:
            self.get_logger().info("Goal not reached within the max number of steps.")
        return trajectory[:step + 2]

    def update_potential_plot(self, trajectory, step):
        self.ax1.clear()
        self.ax2.clear()

        x = np.arange(-5.5, 5.5, self.grid_resolution)
        y = np.arange(-5.5, 5.5, self.grid_resolution)
        X, Y = np.meshgrid(x, y)
        Ut = np.zeros((Y.shape[0], X.shape[1]))

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                q_grid = np.array([X[i, j], Y[i, j]])
                Ut[i, j] = self.repulsive_potential(q_grid, self.obstacles, self.d0, self.k_rep) + \
                           self.attractive_potential(q_grid, self.temp_goal, self.k_att)

        self.ax1.imshow(Ut, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='plasma')
        
        if len(self.drone_path) > 1:
            drone_path_array = np.array(self.drone_path)
            self.ax1.plot(drone_path_array[:, 0], drone_path_array[:, 1], color='white', linewidth=2, label="Drone Path")
            
        
        if self.red_detected is False:
            if trajectory is None:
                trajectory = np.zeros((2, self.max_steps), dtype=int)
            trajectory_array = np.array(trajectory)  # Ensure it's a numpy array
            self.ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], color='green', label='rajectory',linewidth=2)


        #     for traj in self.trajectories:
        #         traj_array = np.array(traj)  # Convert each trajectory to a numpy array
        #         self.ax1.plot(traj_array[:, 0], traj_array[:, 1], color='cyan', alpha=0.5, label='Sampled Trajectory',linewidth = 0.5)

        # # Plot the initial trajectory
        #     if self.initial_trajectory is not None:
        #         initial_trajectory_array = np.array(self.initial_trajectory)  # Ensure it's a numpy array
        #         self.ax1.plot(initial_trajectory_array[:, 0], initial_trajectory_array[:, 1], color='purple', label='Initial Trajectory',linewidth = 0.5)

        #     # Plot the optimal trajectory
        #     if self.optimal_trajectory is not None:
        #         optimal_trajectory_array = np.array(self.optimal_trajectory)  # Ensure it's a numpy array
        #         self.ax1.plot(optimal_trajectory_array[:, 0], optimal_trajectory_array[:, 1], color='red', label='Optimal Trajectory',linewidth=0.5)

        self.ax2.plot_surface(X, Y, Ut, cmap='plasma', edgecolor='black')
        self.ax1.scatter(*self.goal, color='red', s=100, label='Goal', edgecolor='black')
        self.ax1.scatter(*self.start, color='cyan', s=50, label='Start', edgecolor='black')
        self.ax1.scatter(*self.temp_goal, color='red', s=50, label='temp_goal', edgecolor='black')
        self.ax1.scatter(*self.current_position, color='black', s=25, label='drone', edgecolor='white')

        self.fig.canvas.draw()
        plt.pause(0.1)


    def objective_function(self, trajectory, all_trajectories):
        # Convert trajectory to numpy array
        trajectory = np.array(trajectory)
        obstacles = np.array(self.obstacles)

        # Compute total distance (sum of distances between consecutive points)
        diffs = trajectory[1:] - trajectory[:-1]
        distances = np.linalg.norm(diffs, axis=1)
        total_distance = np.sum(distances)

        # Calculate angle deviations between consecutive segments
        angle_deviations = []
        for i in range(1, len(trajectory) - 1):
            vector1 = trajectory[i] - trajectory[i - 1]
            vector2 = trajectory[i + 1] - trajectory[i]
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)

            if norm_vector1 > 0 and norm_vector2 > 0:
                unit_vector1 = vector1 / norm_vector1
                unit_vector2 = vector2 / norm_vector2
                dot_product = np.dot(unit_vector1, unit_vector2)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle_deviation = np.arccos(dot_product)
                angle_deviations.append(abs(angle_deviation))

        total_angle_deviation = np.sum(angle_deviations)

        # Distance from the last point to the temporary goal
        last_point = trajectory[-1]
        distance_to_temp_goal = np.linalg.norm(last_point - self.temp_goal)

        min_distances_to_obstacles = []

        for point in trajectory:
            # For each trajectory point, calculate the distance to each obstacle's center
            distances_to_obstacles = []
            for obs in self.obstacles:
                obstacle_center = np.array(obs[:2])  # The center coordinates of the obstacle
                radius = obs[-1]
                distance_to_obstacle = np.linalg.norm(point - obstacle_center) - radius
                if distance_to_obstacle < 0:
                    distance_to_obstacle = 0.001  # Euclidean distance to the obstacle center
                distances_to_obstacles.append(distance_to_obstacle)
            
            # Find the minimum distance to any obstacle for the current trajectory point
            min_distance_to_obstacle = min(distances_to_obstacles)
            min_distances_to_obstacles.append(min_distance_to_obstacle)

        # Now find the minimum of all the minimum distances (i.e., the closest point to any obstacle in the trajectory)
        total_min_distance_to_obstacles = min(min_distances_to_obstacles) if min_distances_to_obstacles else 0.001
        #self.all_minimum_distances_to_obstacles.append(total_min_distance_to_obstacles)

        # Store each term to calculate the minimum value across all trajectories
        all_distances = []
        all_angle_deviations = []
        all_distance_to_temp_goals = []
        all_min_distances_to_obstacles = []

        for traj in all_trajectories:
            diffs = traj[1:] - traj[:-1]
            distances = np.linalg.norm(diffs, axis=1)
            all_distances.append(np.sum(distances))

            angle_deviations = []
            for i in range(1, len(traj) - 1):
                vector1 = traj[i] - traj[i - 1]
                vector2 = traj[i + 1] - traj[i]
                norm_vector1 = np.linalg.norm(vector1)
                norm_vector2 = np.linalg.norm(vector2)

                if norm_vector1 > 0 and norm_vector2 > 0:
                    unit_vector1 = vector1 / norm_vector1
                    unit_vector2 = vector2 / norm_vector2
                    dot_product = np.dot(unit_vector1, unit_vector2)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle_deviation = np.arccos(dot_product)
                    angle_deviations.append(abs(angle_deviation))

            all_angle_deviations.append(np.sum(angle_deviations))

            last_point = traj[-1]
            all_distance_to_temp_goals.append(np.linalg.norm(last_point - self.temp_goal))

            min_distances_to_obstacles = []
            for point in traj:
                distances_to_obstacles = []
                for obs in self.obstacles:
                    obstacle_center = np.array(obs[:2])
                    radius = obs[-1]
                    distance_to_obstacle = np.linalg.norm(point - obstacle_center) - radius
                    if distance_to_obstacle < 0:
                        distance_to_obstacle = 0.0001
                    distances_to_obstacles.append(distance_to_obstacle)

                min_distance_to_obstacle = min(distances_to_obstacles)
                min_distances_to_obstacles.append(min_distance_to_obstacle)
        

            all_min_distances_to_obstacles.append(min(min_distances_to_obstacles))
        self.get_logger().info(f"final distance_to_obstacle: {total_min_distance_to_obstacles}")

        # Find the minimum value for each term across all trajectories
        min_total_distance = sum(all_distances)
        min_angle_deviation = sum(all_angle_deviations)
        min_distance_to_temp_goal = sum(all_distance_to_temp_goals)
        min_min_distance_to_obstacle = sum(all_min_distances_to_obstacles)

        # Normalize each term by its respective minimum value
        total_distance /= min_total_distance
        total_angle_deviation /= min_angle_deviation
        distance_to_temp_goal /= min_distance_to_temp_goal
        total_min_distance_to_obstacles /= min_min_distance_to_obstacle

        # Log each component for debugging
        self.get_logger().info(f"Normalized Total distance traveled: {total_distance}")
        self.get_logger().info(f"Normalized Total angle deviation: {total_angle_deviation}")
        self.get_logger().info(f"Normalized Distance to temporary goal: {distance_to_temp_goal}")
        self.get_logger().info(f"Normalized Total min distance to obstacles: {total_min_distance_to_obstacles}")

        # Combine into final objective value
        objective_value = (total_distance 
                            + 3*total_angle_deviation 
                            + 2*distance_to_temp_goal 
                            + 5*1/(total_min_distance_to_obstacles))

        return objective_value
    def is_obstacle_in_path(self, start_point, end_point):
        """
        Check if there is an obstacle between the start point and the end point.
        This method checks if any obstacles are within the line segment between these two points.
        """
        for obs in self.obstacles:
            # Calculate the closest distance from the obstacle center to the line segment
            center = np.array(obs[:2])
            radius = obs[2]
            if np.linalg.norm(end_point-center) < radius+self.drone_radius:
                return True
            # Vector from start_point to end_point
            # line_vector = end_point - start_point
            # line_length = np.linalg.norm(line_vector)
            
            # # Normalize the line vector
            # line_unit_vector = line_vector / line_length
            
            # # Vector from start_point to the obstacle center
            # obs_vector = center - start_point
            
            # # Projection of obs_vector onto the line unit vector (i.e., perpendicular distance to the line)
            # projection = np.dot(obs_vector, line_unit_vector)
            
            # # Find the closest point on the line to the obstacle center
            # closest_point_on_line = start_point + projection * line_unit_vector
            
            # # Calculate the distance from the obstacle center to the closest point on the line
            # distance_to_line = np.linalg.norm(center - closest_point_on_line)
            
            # # Check if the obstacle is within a threshold distance from the line segment
            # if distance_to_line < radius and projection >= 0 and projection <= line_length:
            #     return True  # There is an obstacle in the path
        
        return False  # No obstacles in the path

        

    
    def start_trajectory_computation(self):
        start_time = time.time()
        #self.initial_trajectory, self.trajectories, self.optimal_trajectory,self.trajectory = self.optimization()  # Call optimization to get trajectories
        self.trajectory=self.gradient_descent_trajectory_Ut(
            obstacles=self.obstacles,
            d0=self.d0,
            k_att=self.k_att,
            k_rep=self.k_rep
        )
        end_time = time.time()
        computation_time = end_time - start_time
        self.computation_times.append(computation_time)

        self.all_minimum_distance_to_obstacle.append(self.obstacle_distance(self.trajectory))



    def publish_velocity_commands(self):
        """Publish velocity commands to move the drone along the trajectory."""
    # Ensure that the trajectory is generated and has enough points
        if self.trajectory is None or len(self.trajectory) < self.max_steps:
            self.get_logger().info("Waiting for trajectory generation to complete.")
            return  # Wait until trajectory is generated
        self.drone_path.append(self.current_position)
        if self.red_detected is True:
            return

    # Ensure current_index is within bounds of trajectory
        if self.current_index >= len(self.trajectory):
            self.get_logger().info("Reached the end of the trajectory.")
            
            self.trajectory_active = False
            return  # Stop if we've reached the end of the trajectory

    # Calculate error from current position to next point in trajectory
        target_point = self.trajectory[self.current_index]
        #self.get_logger().info(f"Next target point:{target_point}")
        #self.get_logger().info(f"Drone position:{self.current_position}")
        self.publish_point(target_point)
        error = target_point - self.current_position
        #self.get_logger().info(f"Error:{error}")
        
        # error_x = target_point[0] - self.current_position[0]
        
        # error_y = target_point[1] - self.current_position[1]
        error_x = error[0]
        error_y = error[1]
        target_angle = math.atan2(error_y, error_x)
        heading_error = target_angle - self.yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # Normalize angle

    # Yaw PID control
        kp_yaw = 0.5
        ki_yaw = 0.01
        kd_yaw = 0.1
        yaw_derivative = heading_error - self.prev_heading_error
        yaw_velocity = (kp_yaw * heading_error + ki_yaw * self.integral_heading_error + kd_yaw * yaw_derivative)
        self.integral_heading_error += heading_error
        self.prev_heading_error = heading_error
        velocity_command = Twist()
        velocity_command.angular.z = yaw_velocity
        self.velocity_publisher.publish(velocity_command)
        
        distance_to_target = np.linalg.norm(error)
        if distance_to_target > 0.1:  # Avoid division by zero
            self.integral_x += error_x
            self.integral_y += error_y
            derivative_x = error_x - self.prev_error_x
            derivative_y  = error_y  - self.prev_error_y
        #kp = [0.1,0.08]
            kpx = 0.2
            kpy = 0.2
            #kpy = 0.1
            ki = 0.0
            kd = 0.0
            v_global_x = (kpx*error_x+ki*self.integral_x + kd*derivative_x)
            v_global_y = (kpy*error_y+ki*self.integral_y + kd*derivative_y)

            v_drone_x = v_global_x * math.cos(self.yaw) + v_global_y * math.sin(self.yaw)
            v_drone_y = -v_global_x * math.sin(self.yaw) + v_global_y * math.cos(self.yaw)
        else:
            v_drone_x = 0.0
            v_drone_y = 0.0

    # Publish velocity commands
        velocity_command = Twist()
        velocity_command.linear.x = v_drone_x
        velocity_command.linear.y = v_drone_y
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        
        
        
        
        
        #self.get_logger().info(f"Velocity_x:{v_x}, Velocity_y:{v_y}")
        #mag = np.linalg.norm([v_x,v_y])
        
        
        # velocity_command.linear.x = error[1]*0.15
        # velocity_command.linear.y = -error[0]*0.15

        self.velocity_publisher.publish(velocity_command)
        self.drone_path.append(self.current_position)

    # If the drone is close to the target point, move to the next point in trajectory
        if np.linalg.norm(error) < 0.1:
            if tuple(target_point) not in self.reached_points:
                self.reached_points.add(tuple(target_point))
                self.get_logger().info(f"Reached target point {target_point}")
            self.current_index += 1  # Move to the next target in the trajectory
            self.get_logger().info(f"Current index:{self.current_index}")

            


    def check_and_compute_trajectory(self):
        """ Continuously check trajectory progress and update if necessary """
        if not self.trajectory_active:
            self.get_logger().info("Trajectory generation is not active, starting computation...")
            self.trajectory = None  # Clear the previous trajectory
            self.start_trajectory_computation()
            self.current_index = 0
            self.trajectory_active = True
            self.temp_goal_updated = False  # Reset the flag when starting a new trajectory

    # If halfway reached and temp_goal not updated, update and regenerate trajectory
        #if self.current_index >= len(self.trajectory) // 2 and not self.temp_goal_updated:
        if self.current_index >= 5:
            if np.linalg.norm(self.temp_goal-self.goal)>0.01:
                self.update_temp_goal()
            #self.temp_goal_updated = True
            #if not self.temp_goal_updated:
            self.get_logger().info("Reached first point, updating the goal and generating new trajectory.")
                # Update temp goal
            self.trajectory = None  # Clear previous trajectory
            self.start_trajectory_computation()  # Recompute the trajectory
            self.current_index = 0  # Reset to start of new trajectory
            self.trajectory_active = True  # Reactivate the trajectory
                #self.temp_goal_updated = True  # Flag to prevent further updates
            velocity_command = Twist()
            velocity_command.linear.x = 0.0
            velocity_command.linear.y = 0.0
           # self.velocity_publisher.publish(velocity_command)
        self.publish_velocity_commands()
        self.update_potential_plot(self.trajectory, self.current_index)
    def publish_point(self,target):
        msg = Point()
        msg.x = target[0] # Fixed X value (change as needed)
        msg.y = target[1] # Fixed Y value (change as needed)
        #self.get_logger().info(f'Publishing: x={msg.x}, y={msg.y}')
        self.publisher_.publish(msg)
    def listener_callback(self, data):
        frame = self.br.imgmsg_to_cv2(data)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        lower_bound = (0, 70, 10)
        upper_bound = (10, 255, 255)
        mask1 = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
        lower_bound = (170, 70, 10)
        upper_bound = (179, 255, 255)
        mask2 = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
        mask = mask1 + mask2
        num_red_pixels = cv2.countNonZero(mask)
        
        if num_red_pixels >= 500:
            #velocity = Twist()
            #velocity.linear.x = 0.00
            #velocity.linear.y = 0.00
            #velocity.linear.z = 0.00
            #self.velocity_publisher.publish(velocity)
            #self.land_process = subprocess.Popen(["python3", "../landing/cam_control.py"])
           
            self.red_detected = True
    
    def stopping(self):
        end_distance = np.linalg.norm(self.current_position - self.goal)
        if end_distance < 0.05:
            total_distance = self.total_distance()
            average_obstacle_distance = np.mean(self.all_minimum_distance_to_obstacle)
            average_computation_time = np.mean(self.computation_times)

            self.get_logger().info(f"total length of trajectory:{total_distance}")
            self.get_logger().info(f"obstacle average: {average_obstacle_distance}")
            
            for i in range(len(self.computation_times)):
                self.get_logger().info(f"Time taken to compute trajectory {i}th update: {self.computation_times[i]}")
                
            self.get_logger().info(f"Average computation time: {average_computation_time}")
            
            self.simulation_ended = True

            # --- Save to text file ---
            # save_path = "plots/apf/medium_sampled/run_results_apf_medium_sampled.txt"
            # try:
            #     with open(save_path, "r") as f:
            #         lines = f.readlines()
            #     run_number = len(lines) + 1
            # except FileNotFoundError:
            #     run_number = 1

            # with open(save_path, "a") as f:
            #     f.write(f"Run {run_number}: Total Distance = {total_distance:.4f}, "
            #             f"Obstacle Avg = {average_obstacle_distance:.4f}, "
            #             f"Avg Computation Time = {average_computation_time:.4f},"
            #             f"Number of updates = {len(self.computation_times):.4f}\n")
            # self.fig.savefig(f"plots/apf/medium_sampled/apf_medium_sampled_{run_number}_success.png")
            # self.file_saved = True
            # self.get_logger().info(f"everything saved")

    def total_distance(self):
        distances = np.linalg.norm(np.diff(self.drone_path, axis=0), axis=1)  # Compute distances between consecutive points
        return np.sum(distances)
    def obstacle_distance(self,traj):
        min_distances_to_obstacles = []
        #all_min_distances_to_obstacles = []
        for point in traj:
            distances_to_obstacles = []
            for obs in self.obstacles:
                obstacle_center = np.array(obs[:2])
                radius = obs[-1]
                distance_to_obstacle = np.linalg.norm(point - obstacle_center) - radius
                if distance_to_obstacle < 0:
                    distance_to_obstacle = 0.0001
                distances_to_obstacles.append(distance_to_obstacle)

            min_distance_to_obstacle = min(distances_to_obstacles)
            min_distances_to_obstacles.append(min_distance_to_obstacle)
        min_distance_to_closest_obstacle = min(min_distances_to_obstacles)
        return min_distance_to_closest_obstacle
        #average = np.mean(self.all_minimum_distances_to_obstacles)
def get_current_run_number():
    try:
        with open("plots/apf/medium_sampled/run_results_apf_medium_sampled.txt", "r") as f:
            lines = f.readlines()
        return len(lines) + 1
    except FileNotFoundError:
        return 1

        

def main(args=None):
    rclpy.init(args=args)
    apf_node= APF()

    try:
        rclpy.spin(apf_node)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Interrupted — checking run status.")

        if not apf_node.file_saved:
            save_failed_run()
            run_number = get_current_run_number() - 1
            print("Saved: didn't work")
        else:
            run_number = get_current_run_number()


        try:
            rclpy.shutdown()  # Only shut down if not already shut down
        except Exception as e:
            print(f"[Info] rclpy already shutdown: {e}")

    finally:
        apf_node.destroy_node()
        rclpy.shutdown()

        
if __name__ == '__main__':
    main()

