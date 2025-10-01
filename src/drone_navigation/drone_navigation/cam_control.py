   
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge 
import cv2 
import imutils
import numpy as np
from geometry_msgs.msg import Twist
import os
import subprocess
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

class ImageSubscriber(Node):
 
  def __init__(self):
    super().__init__('image_subscriber')
      

    self.subscription = self.create_subscription(Image, '/simple_drone/bottom/image_raw', self.listener_callback, 10)
    #self.subscription_camera_info = self.create_subscription(CameraInfo, '/simple_drone/bottom/camera_info', self.camera_info_callback, 10)
    self.vel_cmd=self.create_publisher(Twist,"/simple_drone/cmd_vel",10)
    self.subscription # prevent unused variable warning
    
    qos_profile = QoSProfile(depth=1)
    qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    self.landing_sub = self.create_subscription(Bool,"/landing",self.landing_callback,qos_profile)

    self.height_pub = self.create_publisher(Bool,"/height",qos_profile)

      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
    self.integral_x = 0.0
    self.integral_y = 0.0
    self.prev_err_x = 0.0
    self.prev_err_y = 0.0
    self.threshold = 10.0
    self.centered_time = 0.0
    self.centered_duration = 3.0  # Duration in seconds to confirm the drone is centered
    self.focal_length = None  # Focal length from the camera info topic
    self.marker_diameter = 0.1
    self.timer = self.create_timer(0.1, self.check_centered)
    self.command_executed = False
    self.height_control_process = None
    self.processing_color = "red"
    self.green_processing_started = False
    self.diameter = 0
    self.landing_initiate = False
    self.call_height = False
    

   
  def landing_callback(self,msg):
      self.landing_initiate = msg.data
      print("Recived msg to land")
 
          
  def listener_callback(self, data):
    if not self.landing_initiate:
        # print("nu uh")
        return
    self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
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
            
    elif self.processing_color == "green":
            lower_bound = (36, 25, 25)
            upper_bound = (86, 255, 255)
            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
            
    #mask_rgb = cv2.cvtColor(mask3,cv2.COLOR_HSV2BGR)
    framep = frame.copy()
    framep[np.where(mask==0)]=0
    #mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    #framep = frame & mask_rgb
    grey = cv2.cvtColor(framep, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (7, 7), 0)
    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    min_contour_area = 100
    desired_x = (frame.shape[1])//2
    desired_y = (frame.shape[0])//2
    frame[desired_y, desired_x]=(255, 0, 0)
    v1 = 0.0
    v2 = 0.0
    found = False
    for c in cnts:
	# compute the center of the contour
         if cv2.contourArea(c) > min_contour_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                

                # Fit a minimum enclosing circle to the largest contour
                (x, y), radius = cv2.minEnclosingCircle(c)

                # Calculate diameter
                self.diameter = radius * 2
                cv2.putText(frame, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                found = True
    if self.diameter >= frame.shape[0]:  # Check if the diameter reaches the height of the frame
        if self.processing_color == "red" and not self.green_processing_started:
                self.processing_color = "green"
                
                self.green_processing_started = True
                return  # Skip the rest of the processing for this frame
    if found:
        err_x = desired_x - cX
        err_y = desired_y - cY
        self.integral_x += err_x
        derivative_x = err_x - self.prev_err_x
        self.integral_y += err_y
        derivative_y = err_y - self.prev_err_y

        kp = 0.0025
        ki = 0.000015
        kd = 0.0024

        v1 = (kp*(err_x)+ki*self.integral_x + kd*derivative_x)
        v2 = (kp*(err_y)+ki*self.integral_y + kd*derivative_y)
        self.prev_err_x = err_x
        self.prev_err_y = err_y

        if abs(err_x) < self.threshold and abs(err_y) < self.threshold:
                self.centered_time += 0.1
        else:
                self.centered_time = 0.0
        if self.processing_color == "green" and self.diameter >= frame.shape[0]-50:
            self.get_logger().info('x-y alignment ended.')
            #cv2.destroyAllWindows()
            rclpy.shutdown()
    else :
        vx = 0.0
        vy = 0.0
        self.centered_time = 0.0
    self.velocity_cmnd(v1,v2)
    self.latest_frame = frame
    # Display image
    cv2.imshow("camera",frame)
    
    cv2.waitKey(1)


  def check_centered(self):
    if not self.landing_initiate:
        # print("check? nu uh")
        return
    if self.centered_time >= self.centered_duration and not self.command_executed:
        self.velocity_cmnd(0.0, 0.0)
        self.get_logger().info('Drone is centered and stable. initiating landing sequence ')
        # self.height_control_process = subprocess.Popen(["python3", "height2.py"])
        self.call_height = True
        msg = Bool()
        msg.data = self.call_height
        self.height_pub.publish(msg)

        # self.height_control_process = subprocess.Popen(["play", "/home/aryan/Downloads/Interstellar Soundtrack- no time for caution but only the best part is in.mp3"])
        self.command_executed = True
        


  def velocity_cmnd(self,v1,v2):
    if self.landing_initiate:
      my_msg = Twist()
      my_msg.linear.x = v2
      my_msg.linear.y = v1
      
      self.vel_cmd.publish(my_msg)
  
  # def camera_info_callback(self, camera_info_msg):
  #   self.focal_length_x = camera_info_msg.k[0]  # Focal length in the x direction
  #   self.focal_length_y = camera_info_msg.k[4]  # Focal length in the y direction

  
  
  # def altitude(self):
  #   #self.get_logger().info('Executing altitude control')
  #   #frame = self.br.imgmsg_to_cv2(data)
  #   frame = self.latest_frame
  #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #   hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    # mask1 = cv2.inRange(hsv, (0, 70, 10), (10, 255, 255))
    # mask2 = cv2.inRange(hsv, (170, 70, 10), (179, 255, 255))
    # mask3 = mask1 + mask2
    # framep = frame.copy()
    # framep[np.where(mask3 == 0)] = 0

    # grey = cv2.cvtColor(framep, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(grey, (7, 7), 0)
    # thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)

    # for c in cnts:
    #     if cv2.contourArea(c) > 100:
    #         ((x, y), radius) = cv2.minEnclosingCircle(c)
    #         if radius > 0:
    #             marker_pixel_diameter = radius * 2
    #             height = (self.marker_diameter * self.focal_length_x) / marker_pixel_diameter
    #             self.get_logger().info(f'Calculated height: {height} meters')
    #             break
    
        
        
def main(args=None):
  
  
  rclpy.init(args=args)
  
  
  image_subscriber = ImageSubscriber()
  
  
  rclpy.spin(image_subscriber)
  

  image_subscriber.destroy_node()
  
  
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
