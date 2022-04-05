from PIL import Image, ImageOps 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import yaml
import pandas as pd
import os

from copy import copy, deepcopy
import time

class Map():
    '''
    Class to load a ROS map
    '''
    def __init__(self, map_name):
        self.res = 0
        self.offs_x,self.offs_y = 0,0
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
    
    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        plt.show()
        return ""
        
    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        #map_name = map_df.image[0]
        file = open(map_name + '.pgm','rb')
        WH = str(file.read(48))
        W = int(WH.split('\\n')[2].split(' ')[0])
        H = int(WH.split('\\n')[2].split(' ')[1])
        im = Image.open(map_name + '.pgm')        
        size = W, H
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        im = im.rotate(90, Image.NEAREST, expand = 1)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        self.res = map_df.resolution[0]
        self.offs_x,self.offs_y = map_df.origin[0][0],map_df.origin[0][1] 
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array

class Queue():
    '''
    Class to implement queues. This class is used to solve the path planning problem
    since it is used to keep track of the nodes that have to be visited the next 
    iteration of the solver.
    '''
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1
    
    def __len__(self):
        numel = len(self.queue)
        return numel
    
    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True
            
            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'
            
        return tmpstr
    
    def __call__(self):
        return self.queue
    
    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)
    
    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)
        
    def push(self,data):
        self.queue.append(data)
        self.end += 1
    
    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p
    
class Node():
    '''
    A node is the simplest element of a tree. This class
    is used by the tree class to create each of the nodes
    of the tree.
    '''
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []
        
    def __repr__(self):
        return self.name
        
    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)
    
class Tree():
    '''
    This tree calss is a set of nodes and edges, where each edge can be weighted
    according to the cost of moving from two nodes.
    '''
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        #self.g_visual = Graph('G')
    
    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual
    
    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name
            
    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False
    
    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)
    
    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and 
            (i < map_array.shape[0]) and 
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value 
    
    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)
        
    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r
                
    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:                    
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left 
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left 
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])                    
        
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm
    
    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m
    
    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array
    
class AStar():
    '''
    This class will find a time optimal path to the desired end position by using the
    euclidean distance as heuristic.
    '''
    def __init__(self,in_tree):
        self.in_tree = in_tree #Initialize in_tree to the graphed map input
        self.q = Queue() #Create an object q of class Queue
        #Initialize distance(cost) to go from node to node to infinity
        self.dist = {name:np.Inf for name,node in in_tree.g.items()}
        #Initialize heuristic cost from node to goal to zero
        self.h = {name:0 for name,node in in_tree.g.items()}
        #Convert the coordinates of the desired goal node to integer
        end_node = in_tree.g[in_tree.end]
        self.end = tuple(map(int, end_node.name.split(',')))

        for name,node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            #Calculate heuristic cost as euclidean distance between each node to goal
            self.h[name] = np.sqrt((self.end[0]-start[0])**2 + (self.end[1]-start[1])**2)

        #Initialize nodes visited to 0
        self.via = {name:0 for name,node in in_tree.g.items()}

        #Setup Queue object, q, to values from graphed map
        # for __,node in in_tree.g.items():
        #     self.q.push(node)

    def __get_f_score(self,node):
        #Returns f score for node as sum of 'total cost to go to node ' and 'heuristic cost'
        return self.dist[node.name] + self.h[node.name]
    
    def solve(self, sn, en):
        self.dist[sn.name] = 0
        self.q.push(sn)
        #While loop runs until queue is empty
        while len(self.q) > 0:
            #Sort Queue in ascending order of f_score at each iteration
            self.q.sort(key=self.__get_f_score)
            #Pops out first value of the queue- one with smallest f_score (total cost)
            u = self.q.pop()
            
            #Exits the loop if goal node has been reached
            if u.name == en.name:
                break

            #Runs loop for all children/neighbors of the extracted node
            for i in range(len(u.children)):
                #Storing the current child and its weight in c and w respectively
                c = u.children[i]
                w = u.weight[i]
                #Calculating new cost as sum of previous cost and current cost
                new_dist = self.dist[u.name] + w
                #Checking that the node has not been visited already and that new cost is less that previous cost
                if new_dist < self.dist[c.name]:
                    #Updating cost and marking node as visited
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name
                    self.q.push(c)

    def reconstruct_path(self,sn,en):
        #Initializing start and end keys from the graphed map
        start_key = sn.name
        end_key = en.name
        #Setting distance as the final cost
        dist = self.dist[end_key]
        #Initializing path to be the end key
        u = end_key
        path = [u]
        #Retracing the path from end to start and appending to path
        while u != start_key:
            u = self.via[u]
            path.append(u)
        #Reversing the path to go from start to end
        path.reverse()
        return path,dist

def angdiff(a, b=None):
        an = np.mod(b-a,2*np.pi)
        if an >= np.pi:
            an -= 2*np.pi
        return an
        

if __name__== '__main__':
    # Load the map and create a graph out of it
    mp = MapProcessor('../maps/my_map') # Load map
    print(mp.map)
    mp.map_graph.root = "0,0"         # Start of the maze (row,col)
    mp.map_graph.end = "10,10"         # End of the maze (row,col)
    kr = mp.rect_kernel(5,1)            # Define how the obstacles will be inflated
    mp.inflate_map(kr,True)             # Inflate map   
    mp.get_graph_from_map()             # Get a graph out of the map
    # Find a path from start to end
    as_maze = AStar(mp.map_graph)
    as_maze.solve(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
    # Get the elements of the path
    path_as,dist_as = as_maze.reconstruct_path(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
    path_arr_as = mp.draw_path(path_as)
    print(path_as)
    fig, ax = plt.subplots(dpi=100)
    plt.imshow(path_arr_as)
    plt.colorbar()
    plt.show()