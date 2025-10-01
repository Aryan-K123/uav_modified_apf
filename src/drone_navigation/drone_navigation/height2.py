import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import imutils
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

class AltitudeControl(Node):

    def __init__(self):
        super().__init__('altitude_control')
        self.subscription             = self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.listener_callback, 10)
        self.subscription_camera_info = self.create_subscription(CameraInfo, '/simple_drone/bottom/camera_info', self.camera_info_callback, 10)
        self.vel_cmd                  = self.create_publisher(Twist,"/simple_drone/cmd_vel",10)
        self.land_publisher           = self.create_publisher(Bool, '/landed', 10)

        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.height_sub               = self.create_subscription(Bool,"/height",self.height_callback,qos_profile)
        self.br = CvBridge()
        self.focal_length_x = 0.0
        
        self.integral_z = 0.0
        self.prev_err_z = 0.0
        self.threshold = 0.02
        self.stable_time = 0.0
        self.stable_duration = 3.0
        #self.timer = self.create_timer(0.1, self.check_stable)
        self.processing_color = "red"
        self.green_processing_started = False
        self.start_height = False
        
    def height_callback(self,msg):
        # Latch behavior: once True, always stay True
        if msg.data:
            self.start_height = True

    def listener_callback(self, data):
        #self.get_logger().info('Receiving video frame for altitude control')
        if not self.start_height:
            return
        frame = self.br.imgmsg_to_cv2(data)
     
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if self.processing_color == "red":
            lower_bound = (0, 70, 10)
            upper_bound = (10, 255, 255)
            mask1 = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
            lower_bound = (170, 70, 10)
            upper_bound = (179, 255, 255)
            mask2 = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
            mask = mask1 + mask2
            marker_diameter = 0.8
        elif self.processing_color == "green":
            lower_bound = (36, 25, 25)
            upper_bound = (86, 255, 255)
            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
            marker_diameter = 0.1
            
        
        framep = frame.copy()
        framep[np.where(mask==0)]=0
        grey = cv2.cvtColor(framep, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (7, 7), 0)
        thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            return
        largest_contour = max(cnts, key=cv2.contourArea)  

        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        diameter = radius * 2
        height = (marker_diameter * self.focal_length_x) / diameter
        cv2.putText(frame,str(height), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame,str(diameter), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if diameter >= frame.shape[0]:  # Check if the diameter reaches the height of the frame
            if self.processing_color == "red" and not self.green_processing_started:
                self.processing_color = "green"
                self.get_logger().info('switching to green color marker')
                self.green_processing_started = True
            return  # Skip the rest of the processing for this frame
             

        cv2.imshow("height camera",frame)
       

        desired_z = 0.0
        err_z = -desired_z + height
        cv2.putText(frame,str(err_z), (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        self.integral_z += err_z
        derivative_z = err_z - self.prev_err_z
        

        kp = 0.2
        ki = 0.000015
        kd =0.0024

        vz = (kp*(err_z)+ki*self.integral_z + kd*derivative_z)
        
        self.prev_err_z = err_z
        if abs(vz) == 0.0:
                self.stable_time += 0.1
        else:
                self.stable_time = 0.0
        self.velocity_cmnd(vz)
        
        cv2.waitKey(1)
        land = False
        if self.processing_color == "green" and diameter >= frame.shape[0]-50:
            self.get_logger().info('landing is ended')
            self.landed_callback()
            land = True
        if land:
            cv2.destroyAllWindows()
            rclpy.shutdown()
            
                                                                                                                                                          
            
    def camera_info_callback(self, camera_info_msg):
        if not self.start_height:
            return
        self.focal_length_x = camera_info_msg.p[0] # Focal length in the x direction
        
    def landed_callback(self):
        if not self.start_height:
            return
        msg = Bool()
        msg.data = True
        self.land_publisher.publish(msg)
        #self.get_logger().info('Publishing: "True"')
        
    def velocity_cmnd(self,vz):
        if self.start_height:   
            my_msg = Twist()
            my_msg.linear.z = -vz
            self.vel_cmd.publish(my_msg)

def main(args=None):
    rclpy.init(args=args)
    altitude_control = AltitudeControl()
    rclpy.spin(altitude_control)
    altitude_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
