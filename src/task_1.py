#!/usr/bin/env python3

import rospy
import numpy as np
from time import monotonic
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class Explorer:
    def __init__(self):
        rospy.init_node("explore_node", anonymous=True)
        rospy.loginfo("Explorer node started")
        # Subscriptions
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        # Publishers
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        # Rate
        self.rate = rospy.Rate(10)

        # Node variables
        self.dists = [0, 0, 0]
        self.error = 0
        self.prevError = 0

        # Main node loop
        rospy.wait_for_message("/scan", LaserScan)
        self.find_first_wall()
        while not rospy.is_shutdown():
            rospy.loginfo("Explorer node running...")
            self.calc_errors()
            self.follow_wall()
            self.rate.sleep()

        rospy.loginfo("Explorer node exited!")

    def scan_callback(self, data):
        self.dists[0] = data.ranges[300]
        self.dists[1] = data.ranges[240]
        self.dists[2] = data.ranges[0]
        rospy.loginfo(
            f"dist 0: {self.dists[0]}, dist 1: {self.dists[1]}, dist front: {self.dists[2]}"
        )

    def find_first_wall(self):
        startTime = monotonic()
        turn90Time = 3 # s
        turn90Speed = 0.5  # rad/s
        goalDist = 0.5  # m
        forwardSpeed = 0.2  # m/s
        cmd_vel = Twist()

        rospy.loginfo("Finding wall")
        # Turn 90 right
        startTime = monotonic()
        while monotonic() - startTime < turn90Time:
            cmd_vel.angular.z = -turn90Speed
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()
        cmd_vel.angular.z = 0
        self.cmd_vel_pub.publish(cmd_vel)
        # Drive straight
        while self.dists[2] > goalDist:
            cmd_vel.linear.x = forwardSpeed
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()
        cmd_vel.linear.x = 0
        self.cmd_vel_pub.publish(cmd_vel)
        # Turn 90 left
        startTime = monotonic()
        while monotonic() - startTime < turn90Time:
            cmd_vel.angular.z = turn90Speed
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()
        cmd_vel.angular.z = 0
        self.cmd_vel_pub.publish(cmd_vel)

    def calc_errors(self):
        goalDist = 0.4  # m

        self.prevError = self.error
        self.error = goalDist - np.mean(self.dists[0:1])

    def follow_wall(self):
        turnP = 5
        turnD = 5
        forwardSpeed = 0.4  # m/s
        frontThresh = 0.6  # m

        cmd_vel = Twist()
        
        if self.dists[2] < frontThresh:
            cmd_vel.linear.x = -forwardSpeed / 10
            cmd_vel.angular.z = 1.6
        else:
            cmd_vel.linear.x = forwardSpeed
            cmd_vel.angular.z = (
                self.error * turnP + (self.error - self.prevError) * turnD
            )
            if cmd_vel.angular.z > 1.2:
                cmd_vel.angular.z = 1.2
            elif cmd_vel.angular.z < -1.2:
                cmd_vel.angular.z = -1.2
        rospy.loginfo(f"x: {cmd_vel.linear.x} z: {cmd_vel.angular.z}")
        self.cmd_vel_pub.publish(cmd_vel)


if __name__ == "__main__":
    try:
        Explorer()
    except rospy.ROSInterruptException:
        pass
