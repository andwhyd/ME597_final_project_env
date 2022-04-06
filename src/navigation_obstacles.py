#!/usr/bin/env python3

import sys
import os
import numpy as np

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float64MultiArray

from nav_utils import *

class Navigation:
    """! Navigation node class.
    This class should server as a template to implement the path planning and 
    path follower components to move the turtlebot from position A to B.
    """
    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        # ROS related variables
        self.node_name = node_name
        self.rate = 0
        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        # Load map
        cwd = os.path.dirname(os.path.abspath(__file__))
        cwd = cwd + '/'
        self.mp = self.load_map(cwd + '../maps/map')
        self.new_goal = False
        self.pos_cov = [0]*36
        self.global_idx = 0
        self.obstacle_in_range = False
        self.avoid_heading = 0
        self.avoid_speed = 0.0

    def init_app(self):
        """! Node intialization.
        @param  None
        @return None.
        """
        # ROS node initilization
        
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(10)
        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.__goal_pose_cbk, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=1)
        rospy.Subscriber('/obstacle_list', Float64MultiArray, self.__obstacles_cbk, queue_size=1)
        # Publishers
        self.path_pub = rospy.Publisher('global_plan', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    
    def __obstacles_cbk(self,data):
        """! Callback to catch the list of obstacles arround the vehicle.
        @param  data    Float64MultiArray object from RVIZ.
        @return None.
        """
        flag = False
        obstacles = data.data
        x,y,d = obstacles
        quaternion = (self.ttbot_pose.pose.orientation.x,
                      self.ttbot_pose.pose.orientation.y,
                      self.ttbot_pose.pose.orientation.z,
                      self.ttbot_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_heading = euler[2]
        if d<6 and np.abs(y)<3.0:
            flag = True
            self.avoid_heading = current_heading - np.sign(y)*0.9*np.pi/2
            self.avoid_speed = 0.6
        else:
            self.avoid_heading = 0
            self.avoid_speed = 0
        self.obstacle_in_range = flag
        rospy.loginfo('is there any obstacle obstacles?:{}'.format(self.obstacle_in_range))


    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.new_goal = True
        rospy.loginfo('goal_pose:{:.4f},{:.4f}'.format(self.goal_pose.pose.position.x,self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        # TODO: MAKE SURE YOUR POSITION ESTIMATE IS GOOD ENOUGH.
        self.ttbot_pose = data.pose
        self.pos_cov = data.pose.covariance
        #rospy.loginfo('ttbot_pose:{:.4f},{:.4f}'.format(self.ttbot_pose.pose.position.x,self.ttbot_pose.pose.position.y))
        #rospy.loginfo('ttbot_pose:{}'.format(self.pos_cov))
    
    def load_map(self,map_name='../maps/my_map'):
        # Load the map
        mp = MapProcessor(map_name)         # Load map
        kr = mp.rect_kernel(10,1)            # Define how the obstacles will be inflated
        mp.inflate_map(kr,True)             # Inflate map   
        mp.get_graph_from_map()             # Get a graph out of the map
        
        return mp
    
    def __pixstr_2_meters(self,list_str_xy):
        path = Path()
        path.header.frame_id = "/map"
        path.header.stamp = rospy.Time.now()
        W,H = self.mp.map.map_im.width,self.mp.map.map_im.height
        res = self.mp.map.res
        offs_x,offs_y = self.mp.map.offs_x,self.mp.map.offs_y
        px,py = int(list_str_xy[0].split(',')[0]),int(list_str_xy[0].split(',')[1])
        old_x = offs_x + (H-px)*res
        old_y = offs_y + (W-py)*res
        for pxy in list_str_xy:
            px,py = int(pxy.split(',')[0]),int(pxy.split(',')[1])
            x = offs_x + (H-px)*res
            y = offs_y + (W-py)*res
            yaw = np.arctan2(y-old_y,x-old_x)
            old_x,old_y = x,y
            q = tf.transformations.quaternion_from_euler(0,0,yaw)
            pose_s = PoseStamped()
            pose_s.pose.position.x = x
            pose_s.pose.position.y = y
            pose_s.pose.orientation.x = q[0]
            pose_s.pose.orientation.y = q[1]
            pose_s.pose.orientation.z = q[2]
            pose_s.pose.orientation.w = q[3]
            path.poses.append(pose_s)
        return path
    
    def __meters_2_pixstr(self,x,y):
        W,H = self.mp.map.map_im.width,self.mp.map.map_im.height
        res = self.mp.map.res
        offs_x,offs_y = self.mp.map.offs_x,self.mp.map.offs_y
        px = int(H - (x-offs_x) / res)
        py = int(W - (y-offs_y) / res)
        return '{},{}'.format(px,py)
         
    def a_star_path_planner(self,start_pose,end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        rospy.loginfo('A* planner.\n> start:{},\n> end:{}'.format(start_pose.pose.position,end_pose.pose.position))
        # Find a path from start to end
        root_xy = self.__meters_2_pixstr(start_pose.pose.position.x,start_pose.pose.position.y) 
        end_xy  = self.__meters_2_pixstr(end_pose.pose.position.x,end_pose.pose.position.y)  
        self.mp.map_graph.root = root_xy            # Start of the maze (row,col)
        self.mp.map_graph.end = end_xy              # End of the maze (row,col)
        try:
            as_maze = AStar(self.mp.map_graph)
            as_maze.solve(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
            # Get the elements of the path
            path_as,dist_as = as_maze.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
            path_arr_as = self.mp.draw_path(path_as)
            path = self.__pixstr_2_meters(path_as)
            path.poses[-1].pose.orientation = end_pose.pose.orientation
            self.path_pub.publish(path)
        except:
            rospy.loginfo('Path not found')
            path = None
            
        # fig, ax = plt.subplots(dpi=100)
        # plt.imshow(path_arr_as)
        # plt.colorbar()
        # plt.show()
        
        # path.poses.append(start_pose.pose.position)
        
        
        return path
    
    def get_path_idx(self,path,vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position int the path pointing to the next goal pose to follow.
        """
        rg = 4     # Rage to look for optimal position to drive towards
        vx,vy = vehicle_pose.pose.position.x,vehicle_pose.pose.position.y
        p1 = np.array([vx,vy])
        start = self.global_idx
        end = start + rg
        end = end if end < len(path.poses) else len(path.poses)-1
        end = len(path.poses)-1
        min_d = 100000
        # Look for the best point to follow inside the path
        idx = start
        for i,ps in enumerate(path.poses[start:end]):
            x,y = ps.pose.position.x,ps.pose.position.y
            p2 = np.array([x,y])
            d = np.linalg.norm(p1-p2)
            if d < min_d:
                min_d = d
                idx = start + i
                idx = idx if idx < len(path.poses) else len(path.poses) - 1
        # rospy.loginfo('{},{},{}'.format(start,idx,end))
        return idx

    def path_follower(self,vehicle_pose, path, idx):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  path                   Planned path.
        @param  idx                    Index pointing to the current goal waypoint.
        @return speed                  Desired speed to get to the target waypoint
        @return heading                Desired heading to get to the target waypoint
        """
        idx = idx if idx < len(path.poses) else len(path.poses)-1
        current_goal_pose = path.poses[idx]
        p1 = np.array([vehicle_pose.pose.position.x,vehicle_pose.pose.position.y])
        p2 = np.array([current_goal_pose.pose.position.x,current_goal_pose.pose.position.y])
        d = np.linalg.norm(p2-p1)
        quaternion = (current_goal_pose.pose.orientation.x,
                      current_goal_pose.pose.orientation.y,
                      current_goal_pose.pose.orientation.z,
                      current_goal_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        tgt_heading = euler[2] + self.avoid_heading
        if idx < (len(path.poses)-1):
            angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
            delta = angdiff(tgt_heading,angle)
            speed = 0.9*d if self.obstacle_in_range == False else self.avoid_speed
            heading = (tgt_heading + 0.5*delta)
            #rospy.loginfo('tgt:{:.3f},angle:{:.3f}'.format(np.rad2deg(tgt_heading),np.rad2deg(angle)))
        else:
            speed = 0
            heading = tgt_heading
        return speed,heading

    def move_ttbot(self,speed,heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired yaw angle.
        @param  heading   Desired speed.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        quaternion = (self.ttbot_pose.pose.orientation.x,
                      self.ttbot_pose.pose.orientation.y,
                      self.ttbot_pose.pose.orientation.z,
                      self.ttbot_pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        current_heading = euler[2]
        ang_error = angdiff(current_heading,heading)
        cmd_vel.linear.x = speed*(1-np.abs(ang_error)/np.pi)**4
        cmd_vel.angular.z = 0.3*(ang_error)
        # if self.obstacle_in_range == True:
        #     cmd_vel.linear.x = 0
        #     cmd_vel.angular.z = 0
        self.cmd_vel_pub.publish(cmd_vel)
    
    def improve_pos_estimate(self):
        """! This an initial routine to move the robot through the world to improve the position
        estimate.
        @param  None.
        @return None.
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = 0.3

        self.cmd_vel_pub.publish(cmd_vel)
    
    def stop(self):
        """! Simple function to stop the turtlebot.
        @param  None.
        @return None.
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        
        '''
            Main loop
        '''
        path_complete = False
        timeout = False
        idx = 0
        path = None
        error = 10

        # Initial routine to improve vehicle localization
        while(error >= 0.02) or (error == 0):
            self.improve_pos_estimate()
            error = np.abs(self.pos_cov[0])+np.abs(self.pos_cov[7])+np.abs(self.pos_cov[35])
            rospy.loginfo('Error: {}'.format(error))
        rospy.loginfo('Position estimate accuracy reached: {}'.format(error))
        self.stop()        
        # Main loop
        while not rospy.is_shutdown():
            if self.new_goal == True:
                self.new_goal = False
                # 1. Create the path to follow
                self.global_idx = 0
                path = self.a_star_path_planner(self.ttbot_pose,self.goal_pose)
            if path != None:
                # 2. Loop through the path and move the robot
                self.global_idx = self.get_path_idx(path,self.ttbot_pose)
                look_ahead = 5
                speed,heading = self.path_follower(self.ttbot_pose,path,self.global_idx+look_ahead)
                #rospy.loginfo('{:3d},{:.3f},{:.2f}'.format(self.global_idx,speed,heading*180/3.1416))
                self.move_ttbot(speed,heading)
            self.rate.sleep() 
        rospy.signal_shutdown("[{}] Finished Cleanly".format(self.name))


if __name__ == "__main__":
    nav = Navigation(node_name='Navigation')
    nav.init_app()
    try:
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)