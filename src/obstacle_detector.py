#!/usr/bin/env python3


import rospy
from std_msgs.msg import Header

from sensor_msgs.msg import Image,NavSatFix,Imu
from nav_msgs.msg import Odometry
import argparse
import cv2
import numpy as np

from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError
import tf
import math

class Node():
    def __init__(self):
        rospy.loginfo("Init camera node!!")
        self.rate = rospy.Rate(5) # ROS Rate at 5Hz
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.odom_sub = rospy.Subscriber("/odom",Odometry,self.position_callback)
        self.obstacle_pub = rospy.Publisher('/obstacle_list', Float64MultiArray, queue_size=10)
        self.bridge_object = CvBridge()
        self.cv_image = None
        self.rgb_objects_tmp = []
        self.robot_heading = 0
        self.robot_x = 0
        self.robot_y = 0

    def position_callback(self,data):
        """
         Get robot's pose in global frame
         """
        pose = data.pose.pose
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z,
             pose.orientation.w]
        self.robot_x = pose.position.x
        self.robot_y = pose.position.y
        _, _, self.robot_heading = tf.transformations.euler_from_quaternion(q)
    def camera_callback(self, data):
        """
        Detect the moving obstacles and localize the closest one in global frame
        """
        try:
            # We select bgr8 because its the OpneCV encoding by default
            self.cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        k = 5  # set the ratio of resized image
        min_blob_threshold = 40000
        min_area = 500   # set the minimum area
        raw_height, raw_width, channels = self.cv_image.shape
        height = int(raw_height / k)
        width = int((raw_width) / k)
        # resize the image by resize() function of openCV library
        scaled = cv2.resize(self.cv_image, (width, height), interpolation=cv2.INTER_AREA)
        hsv_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([51, 0, 255])
        mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
        res_white = cv2.bitwise_and(scaled, scaled, mask=mask_white)
        # blob detection
        m_white = cv2.moments(mask_white, False)
        try:
            blob_area_white = m_white['m00']
        except ZeroDivisionError:
            blob_area_white = 0
        # print("blob_area_white:", blob_area_white)  # min: 11730    can see : 448035
        if blob_area_white >= min_blob_threshold:
            blur_white = cv2.GaussianBlur(mask_white, (3, 3), 0)  # reduce noise
            thresh_white = cv2.threshold(blur_white, 45, 255, cv2.THRESH_BINARY)[1]  # To clear the yellow line
            blur_white = cv2.erode(thresh_white, None, iterations=1)
            # find contours in thresholded image, then grab the largest one
            contours = cv2.findContours(blur_white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours = cv2.findContours(edged_white_L.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            real_num_cnt = 0
            if len(contours) != 0:
                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])
                    area_old = 0
                    # If contour area bigger than threshold: draw the contour
                    if area > min_area and area>area_old: # keep the one with the largest area
                        self.rgb_objects_tmp = []
                        cv2.drawContours(scaled, contours, i, (0, 230, 255), 5)  # draw the yellow contour
                        real_num_cnt = real_num_cnt + 1
                        # compute the center of the contour
                        M = cv2.moments(contours[i])
                        cX = int(M["m10"] / M["m00"])
                        alpha = -0.003128585*(cX-width/2)
                        dist = 171 * area**(-0.51)
                        # calculate the global position
                        heading = self.robot_heading+alpha
                        x_g = self.robot_x+math.cos(heading)*dist
                        y_g = self.robot_y + math.sin(heading) * dist
                        x_r =  math.cos(alpha) * dist
                        y_r =  math.sin(alpha) * dist
                        self.rgb_objects_tmp.append([round(x_r,2),round(y_r,2) ,round(dist,2)])
                        area_old = area

        if len(self.rgb_objects_tmp)>0:
            #print(self.rgb_objects_tmp)
            obstacle = Float64MultiArray()
            obstacle.data = self.rgb_objects_tmp[0]
            self.obstacle_pub.publish(obstacle)  # publish the position of the obstacle and the distance
        self.rgb_objects_tmp = []  # reset the list
        cv2.imshow("White", res_white)
        cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('lidar_localization_with_rgb')
    node = Node()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
