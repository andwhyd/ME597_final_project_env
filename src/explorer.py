#!/usr/bin/env python3

import rospy

if __name__ == '__main__':
    rospy.init_node('explore_node', anonymous=True)
    
    rospy.loginfo('This is a test!')
    
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        rospy.loginfo('Explorer node alive ...')
        r.sleep()
        
    rospy.loginfo('Explorer node exited!')
