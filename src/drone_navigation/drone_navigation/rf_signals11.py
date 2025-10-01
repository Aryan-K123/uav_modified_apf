import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Twist
from scipy.fft import fft
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess
import matplotlib.pyplot as plt

class RFSignalNode(Node):
    def __init__(self):
        super().__init__('rf_signal_node')

        # Parameters
        self.t0 = 0.0  # Initial time
        self.d = 0.05  # Distance between antennas
        self.l = 0.05  # Length of antennas
        self.wave_source_position = (-8.0, -1.41061646268, 0.0)  # RF source position
        self.w = 2 * np.pi * 5e9  # Frequency in Hz (5 GHz)
        self.k = 2 * np.pi / 0.125  # Wave number for a wavelength of 12.5 cm
        self.Phi = 0  # Phase offset
        self.A0 = 1  # Amplitude (A0)
        self.land_process = None
        self.br = CvBridge()
        self.trajectory = []  # To store the drone's trajectory

        # Sample signal generation parameters
        self.sample_rate = 100e9  # Sample rate (Hz)
        self.duration = 1e-9      # Duration of the signal in seconds
        self.samples = int(self.sample_rate * self.duration)
        self.time = np.linspace(0, self.duration, self.samples, endpoint=False)

        # Pre-generate the signal
        self.stored_signal = self.generate_signal(self.wave_source_position)

        # Subscriptions and publishers
        self.subscription = self.create_subscription(
            Odometry,
            '/simple_drone/odom',
            self.odom_callback,
            10
        )
        self.subscription = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.listener_callback,
            10
        )
        self.direction_publisher = self.create_publisher(Vector3, 'rf_direction', 10)
        self.vel_cmd = self.create_publisher(Twist, "/simple_drone/cmd_vel", 10)

    def generate_signal(self, antenna_position):
        x0, y0, z0 = self.wave_source_position
        x, y, z = antenna_position
        di = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        signal = self.A0 * np.sin(self.k * di - self.w * self.time + self.Phi)
        return signal

    def apply_fft(self, signal):
        Y = fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        return Y, freqs

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        yaw = self.get_yaw_from_quaternion(msg.pose.pose.orientation)
        
        # Store position for trajectory plotting
        self.trajectory.append((x, y))

        # Calculate antenna positions
        antenna_positions = [
            self.calculate_antenna_position(x, y, z, yaw, -self.d/2, -self.d/2),
            self.calculate_antenna_position(x, y, z, yaw, self.d/2, -self.d/2),
            self.calculate_antenna_position(x, y, z, yaw, -self.d/2, self.d/2),
            self.calculate_antenna_position(x, y, z, yaw, self.d/2, self.d/2)
        ]

        # Generate waveforms and perform FFT for each antenna position
        # Use stored signal and shift it according to antenna position
        fft_results = []
        for pos in antenna_positions:
            signal = self.generate_signal(pos)
            Y, freqs = self.apply_fft(signal)
            fft_results.append((Y, freqs))
        
        # Extract phases of the dominant frequency component
        phases = [np.angle(fft_result[0][np.argmax(np.abs(fft_result[0]))]) for fft_result in fft_results]

        # Calculate phase differences
        phase_diff_AB = self.calculate_phase_difference(phases[0], phases[1])
        phase_diff_CD = self.calculate_phase_difference(phases[2], phases[3])
        
        # Calculate angles of arrival
        angle_AB = self.calculate_angle_of_arrival(phase_diff_AB)
        angle_CD = self.calculate_angle_of_arrival(phase_diff_CD)

        # Determine direction vector based on quadrant logic
        direction_vector = self.calculate_direction_vector(angle_AB, angle_CD, yaw)

        # Normalize the direction vector
        direction_vector = self.normalize_vector(direction_vector)
        
        # Publish direction vector and command velocity
        self.publish_direction_vector(direction_vector)
        self.publish_command_velocity(direction_vector)
        
        # Debugging info
        self.get_logger().info(f'phase Values: A={phases[0]}, B={phases[1]}, C={phases[2]}, D={phases[3]}')
        self.get_logger().info(f'Phase Differences: AB={phase_diff_AB}, CD={phase_diff_CD}')
        self.get_logger().info(f'Angles of Arrival: AB={angle_AB}, CD={angle_CD}')
        self.get_logger().info(f'Direction Vector: {direction_vector}')
        
        self.time += 0.1  # Increment time

    def calculate_antenna_position(self, x, y, z, yaw, dx, dy):
        x_antenna = x + dx * np.cos(yaw) - dy * np.sin(yaw)
        y_antenna = y + dx * np.sin(yaw) + dy * np.cos(yaw)
        z_antenna = z + self.l
        return (x_antenna, y_antenna, z_antenna)
    
    def calculate_phase_difference(self, phase1, phase2):
        return np.arctan2(np.sin(phase2 - phase1), np.cos(phase2 - phase1))

    def calculate_angle_of_arrival(self, phase_diff):
        try:
            angle = np.arcsin(phase_diff / (self.k * self.d))
        except ValueError:
            # Handle out-of-bounds values for arcsin
            angle = 0.0
        return angle

    def calculate_direction_vector(self, angle_AB, angle_CD, yaw):
        # Decide which quadrant the signal is coming from based on the AoA
        
        if angle_AB < angle_CD:           
            if -np.pi/2 <= angle_AB <= 0:
                direction_x = -abs(np.sin(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
                direction_y = -abs(np.cos(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
            else:
                direction_x = abs(np.sin(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
                direction_y = abs(np.cos(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
            
        else:
            if -np.pi/2 <= angle_AB <= 0:
                direction_x = -abs(np.sin(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
                direction_y = abs(np.cos(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
            else:
                direction_x = abs(np.sin(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
                direction_y = -abs(np.cos(np.arctan(1/(((1/np.tan(angle_AB)) + (1/np.tan(angle_CD))) / 2)) + yaw))
        
        return Vector3(x=direction_x, y=direction_y, z=0.0)

    def normalize_vector(self, vector):
        magnitude = np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
        if magnitude > 0:
            vector.x /= magnitude
            vector.y /= magnitude
            vector.z /= magnitude
        return vector

    def publish_direction_vector(self, direction_vector):
        self.direction_publisher.publish(direction_vector)

    def get_yaw_from_quaternion(self, q):
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
        
    def publish_command_velocity(self, cmd_velocity):
        velocity = Twist()
        velocity.linear.x = (cmd_velocity.x)
        velocity.linear.y = (cmd_velocity.y)
        velocity.linear.z = (cmd_velocity.z)
        self.vel_cmd.publish(velocity)

    def listener_callback(self, msg):
        image = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Image processing logic here
        pass

    def destroy_node(self):
        super().destroy_node()
        
        # Plot trajectory when node is destroyed
        self.plot_trajectory()

    def plot_trajectory(self):
        # Plot the stored trajectory
        if self.trajectory:
            trajectory_np = np.array(self.trajectory)
            plt.plot(trajectory_np[:, 0], trajectory_np[:, 1], marker='o',markersize=4,linewidth=1, label='Drone Trajectory')

            # Plot the starting point and RF source
            start_point = self.trajectory[0]
            rf_source = self.wave_source_position[:2]  # Extracting only x and y
            plt.plot([start_point[0], rf_source[0]], [start_point[1], rf_source[1]], 
                     linestyle='--', color='red', label='Start to RF Source')

            # Mark the RF source and starting point
            plt.scatter(*start_point, color='green', label='Start Point')
            plt.scatter(*rf_source, color='red', label='RF Source')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Drone Trajectory with RF Source')
            plt.legend()
            plt.grid()
            plt.show()


def main(args=None):
    rclpy.init(args=args)
    rf_signal_node = RFSignalNode()
    try:
        rclpy.spin(rf_signal_node)
    except KeyboardInterrupt:
        pass
    finally:
        rf_signal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

