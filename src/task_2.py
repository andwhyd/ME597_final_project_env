#!/usr/bin/env python3

import sys
import yaml
import rospy
import time
import rospkg
import numpy as np
import pandas as pd
from copy import copy
from graphviz import Graph
from nav_msgs.msg import Path
from PIL import Image, ImageOps
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

import cProfile


class Map:
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __open_map(self, map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + ".yaml", "r")
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        map_name = map_df.image[0]
        im = Image.open(map_name)
        size = 384, 608
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin, xmax, ymin, ymax]

    def __get_obstacle_map(self, map_im, map_df):
        img_array = np.reshape(
            list(self.map_im.getdata()), (self.map_im.size[1], self.map_im.size[0])
        )
        up_thresh = self.map_df.occupied_thresh[0] * 255
        low_thresh = self.map_df.free_thresh[0] * 255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i, j] > up_thresh:
                    img_array[i, j] = 255
                else:
                    img_array[i, j] = 0
        return img_array


class Queue:
    def __init__(self, init_queue=[]):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue) - 1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if i == self.start:
                tmpstr += "<"
                flag = True
            if i == self.end:
                tmpstr += ">"
                flag = True

            if flag:
                tmpstr += "| " + str(q[i]) + "|\n"
            else:
                tmpstr += " | " + str(q[i]) + "|\n"

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self, init_queue=[]):
        self.queue = copy(init_queue)

    def sort(self, key=str.lower):
        self.queue = sorted(self.queue, key=key)

    def push(self, data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue) - 1
        return p


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self, node, w=None):
        if w == None:
            w = [1] * len(node)
        self.children.extend(node)
        self.weight.extend(w)


class Tree:
    def __init__(self, name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph("G")

    def __call__(self):
        for name, node in self.g.items():
            if self.root == name:
                self.g_visual.node(name, name, color="red")
            elif self.end == name:
                self.g_visual.node(name, name, color="blue")
            else:
                self.g_visual.node(name, name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                # print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name, c.name)
                else:
                    self.g_visual.edge(name, c.name, label=str(w))
        return self.g_visual

    def add_node(self, node, start=False, end=False):
        self.g[node.name] = node
        if start:
            self.root = node.name
        elif end:
            self.end = node.name

    def set_as_root(self, node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self, node):
        # These are exclusive conditions
        self.root = False
        self.end = True


class AStar:
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name: np.Inf for name, node in in_tree.g.items()}
        self.h = {name: 0 for name, node in in_tree.g.items()}

        for name, node in in_tree.g.items():
            start = tuple(map(int, name.split(",")))
            end = tuple(map(int, self.in_tree.end.split(",")))
            self.h[name] = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        self.via = {name: 0 for name, node in in_tree.g.items()}
        for __, node in in_tree.g.items():
            self.q.push(node)

    def __get_f_score(self, node):
        return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        while len(self.q) > 0:
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()
            # print(u.name,self.q.queue)
            if u.name == en.name:
                break
            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name

    def reconstruct_path(self, sn, en):
        start_key = sn.name
        end_key = en.name
        dist = self.dist[end_key]
        u = end_key
        path = [u]
        while u != start_key:
            u = self.via[u]
            path.append(u)
        path.reverse()
        return path, dist


class MapProcessor:
    def __init__(self, name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self, map_array, i, j, value, absolute):
        if (
            (i >= 0)
            and (i < map_array.shape[0])
            and (j >= 0)
            and (j < map_array.shape[1])
        ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self, kernel, map_array, i, j, absolute):
        dx = int(kernel.shape[0] // 2)
        dy = int(kernel.shape[1] // 2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array, i, j, kernel[0][0], absolute)
        else:
            for k in range(i - dx, i + dx):
                for l in range(j - dy, j + dy):
                    self.__modify_map_pixel(
                        map_array, k, l, kernel[k - i + dx][l - j + dy], absolute
                    )

    def inflate_map(self, kernel, absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(
                        kernel, self.inf_map_img_array, i, j, absolute
                    )
        r = np.max(self.inf_map_img_array) - np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (
            self.inf_map_img_array - np.min(self.inf_map_img_array)
        ) / r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node("%d,%d" % (i, j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if i > 0:
                        if self.inf_map_img_array[i - 1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g["%d,%d" % (i - 1, j)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_up], [1]
                            )
                    if i < (self.map.image_array.shape[0] - 1):
                        if self.inf_map_img_array[i + 1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g["%d,%d" % (i + 1, j)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_dw], [1]
                            )
                    if j > 0:
                        if self.inf_map_img_array[i][j - 1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g["%d,%d" % (i, j - 1)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_lf], [1]
                            )
                    if j < (self.map.image_array.shape[1] - 1):
                        if self.inf_map_img_array[i][j + 1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g["%d,%d" % (i, j + 1)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_rg], [1]
                            )
                    if (i > 0) and (j > 0):
                        if self.inf_map_img_array[i - 1][j - 1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g["%d,%d" % (i - 1, j - 1)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_up_lf], [np.sqrt(2)]
                            )
                    if (i > 0) and (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i - 1][j + 1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g["%d,%d" % (i - 1, j + 1)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_up_rg], [np.sqrt(2)]
                            )
                    if (i < (self.map.image_array.shape[0] - 1)) and (j > 0):
                        if self.inf_map_img_array[i + 1][j - 1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g["%d,%d" % (i + 1, j - 1)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_dw_lf], [np.sqrt(2)]
                            )
                    if (i < (self.map.image_array.shape[0] - 1)) and (
                        j < (self.map.image_array.shape[1] - 1)
                    ):
                        if self.inf_map_img_array[i + 1][j + 1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g["%d,%d" % (i + 1, j + 1)]
                            self.map_graph.g["%d,%d" % (i, j)].add_children(
                                [child_dw_rg], [np.sqrt(2)]
                            )

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size : size + 1, -size : size + 1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
        r = np.max(g) - np.min(g)
        sm = (g - np.min(g)) * 1 / r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size, size))
        return m

    def draw_path(self, path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(",")))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array


class Navigation:
    """! Navigation node class.
    This class should server as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name="Navigation"):
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

    def init_app(self):
        """! Node intialization.
        @param  None
        @return None.
        """
        # ROS node initilization

        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(10)
        # Subscribers
        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.__goal_pose_cbk, queue_size=1
        )
        rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self.__ttbot_pose_cbk, queue_size=1
        )
        # Publishers
        ros_path = rospkg.RosPack()
        self.map_path = ros_path.get_path("final_project")
        self.path_pub = rospy.Publisher("global_plan", Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        self.mp = MapProcessor(self.map_path + "/maps/map")
        kr = self.mp.rect_kernel(8, 1)
        # kr = mp.rect_kernel(1,1)
        self.mp.inflate_map(kr, True)
        self.mp.get_graph_from_map()

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        rospy.loginfo(
            "goal_pose:{:.4f},{:.4f}".format(
                self.goal_pose.pose.position.x, self.goal_pose.pose.position.y
            )
        )

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        threshold = 0.5
        cov = data.pose.covariance
        if cov[0] < threshold and cov[7] < threshold:
            self.ttbot_pose = data.pose
            rospy.loginfo(
                "ttbot_pose:{:.4f},{:.4f}".format(
                    self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y
                )
            )
            rospy.loginfo("ttbot_pose:{}".format(cov))
        else:
            rospy.loginfo("ttbot_pose:{},{} > threshold".format(cov[0], cov[7]))

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        rospy.loginfo(
            "A* planner.\n> start:{},\n> end:{}".format(
                start_pose.pose.position, end_pose.pose.position
            )
        )
        img_size_h = 243
        img_size_w = 243
        resolution = 0.05
        origin_offset_x = 10
        origin_offset_y = 10

        x_img = (
            abs(round(start_pose.pose.position.y / resolution - img_size_h / 2))
            - origin_offset_x
        )
        y_img = abs(round(start_pose.pose.position.x / resolution - img_size_w / 2))
        if y_img < (img_size_w / 2) - origin_offset_y:
            y_img = round(img_size_w / 2 + abs(img_size_w / 2 - y_img)) - origin_offset_y

        elif y_img > (img_size_w / 2) + origin_offset_y:
            y_img = round(img_size_w / 2 - abs(img_size_w / 2 - y_img)) - origin_offset_y
        self.mp.map_graph.root = f"{x_img},{y_img}"
        rospy.loginfo(f"root = {x_img}, {y_img}")

        x_img = (
            abs(round(end_pose.pose.position.y / resolution - img_size_h / 2))
            - origin_offset_x
        )
        y_img = abs(round(end_pose.pose.position.x / resolution - img_size_w / 2))
        if y_img < (img_size_w / 2) - origin_offset_y:
            y_img = round(img_size_w / 2 + abs(img_size_w / 2 - y_img)) - origin_offset_y

        elif y_img > (img_size_w / 2) + origin_offset_y:
            y_img = round(img_size_w / 2 - abs(img_size_w / 2 - y_img)) - origin_offset_y
        self.mp.map_graph.end = f"{x_img},{y_img}"
        rospy.loginfo(f"end = {x_img}, {y_img}")

        as_maze = AStar(self.mp.map_graph)
        path.poses.append(start_pose)

        start = time.time()
        as_maze.solve(
            self.mp.map_graph.g[self.mp.map_graph.root],
            self.mp.map_graph.g[self.mp.map_graph.end],
        )
        end = time.time()
        print("Elapsed Time: %.3f" % (end - start))

        path_as, _ = as_maze.reconstruct_path(
            self.mp.map_graph.g[self.mp.map_graph.root],
            self.mp.map_graph.g[self.mp.map_graph.end],
        )
        for pose in path_as:
            path_pose = PoseStamped()
            pose = tuple(map(int, pose.split(",")))
            path_pose.pose.position.x = (
                (pose[1] - origin_offset_x) - img_size_h / 2
            ) * resolution
            path_pose.pose.position.y = (
                -((pose[0] + origin_offset_y) - img_size_w / 2) * resolution
            )
            path.poses.append(path_pose)
        path.poses.append(end_pose)
        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose          PoseStamped object containing the current vehicle position.
        @return idx                   Position int the path pointing to the next goal pose to follow.
        """
        vectors = np.empty([len(path.poses), 2])
        for j in range(len(path.poses)):
            vectors[j] = (
                (path.poses[j].pose.position.x - vehicle_pose.pose.position.x),
                (path.poses[j].pose.position.y - vehicle_pose.pose.position.y),
            )
        dists = np.linalg.norm(vectors, axis=1)
        return np.argmin(dists)

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        """
        speed = 0
        heading = 0

        current_x = vehicle_pose.pose.position.x
        current_y = vehicle_pose.pose.position.y
        goal_x = current_goal_pose.pose.position.x
        goal_y = current_goal_pose.pose.position.y

        # Calcuate speed
        dist = np.linalg.norm(((current_x - goal_x), (current_y - goal_y)))
        speed = dist  # "speed"
        # Calculate heading
        if goal_x - current_x == 0:
            heading = 0  # going straight
        else:
            angle_between_points = np.arctan2(goal_y - current_y, goal_x - current_x)
            quat = vehicle_pose.pose.orientation
            _, _, current_heading = euler_from_quaternion(
                [quat.x, quat.y, quat.z, quat.w]
            )
            heading = angle_between_points - current_heading

        rospy.loginfo(f"dist: {speed} angle: {heading}")

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  dist     Desired speed.
        @param  heading   Desired yaw angle.
        """
        cmd_vel = Twist()

        heading_thres = 0.3
        kp_x = 0.8
        kp_z = 0.5

        cmd_vel.linear.x = speed * kp_x
        cmd_vel.angular.z = heading * kp_z

        # Prioritize turning
        if abs(heading) > heading_thres:
            cmd_vel.linear.x = cmd_vel.linear.x / 5

        rospy.loginfo(f"x: {cmd_vel.linear.x} z: {cmd_vel.angular.z}")

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """

        """
            Main loop
        """
        dist_reached = True
        current_goal_pose = self.goal_pose
        current_goal = self.goal_pose
        idx = 0
        dist_threshold = 0.1
        end_threshold = 0.05
        while not rospy.is_shutdown():
            # 1. Create the path to follow
            if current_goal_pose != self.goal_pose:
                self.move_ttbot(0, 0)
                path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                # self.path_pub.publish(path)
                current_goal_pose = self.goal_pose
                dist_reached = False
            # 2. Loop through the path and move the robot
            if not dist_reached:
                idx = self.get_path_idx(path, self.ttbot_pose)
                current_goal = path.poses[idx]
                rospy.loginfo(f"path len = {len(path.poses)}")
                speed, heading = self.path_follower(self.ttbot_pose, current_goal)
                self.move_ttbot(speed, heading)
                dist_current = np.linalg.norm(
                    (
                        (
                            current_goal.pose.position.x
                            - self.ttbot_pose.pose.position.x
                        ),
                        (
                            current_goal.pose.position.y
                            - self.ttbot_pose.pose.position.y
                        ),
                    )
                )
                dist_goal = np.linalg.norm(
                    (
                        (
                            self.goal_pose.pose.position.x
                            - self.ttbot_pose.pose.position.x
                        ),
                        (
                            self.goal_pose.pose.position.y
                            - self.ttbot_pose.pose.position.y
                        ),
                    )
                )
                rospy.loginfo(f"goal current = {dist_current}")
                rospy.loginfo(f"goal dist = {dist_goal}")
                if len(path.poses) > 1 and dist_current < dist_threshold:
                    path.poses.pop(idx)
                if dist_goal < end_threshold:
                    dist_reached = True
            else:
                # _, heading = self.path_follower(self.ttbot_pose, current_goal)
                self.move_ttbot(0, 0)  # heading)
            self.rate.sleep()
        rospy.signal_shutdown(f"[{self.name}] Finished Cleanly")


if __name__ == "__main__":
    nav = Navigation(node_name="Navigation")
    nav.init_app()
    try:
        # cProfile.run("nav.run()")
        nav.run()
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)
