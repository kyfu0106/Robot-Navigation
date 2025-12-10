import matplotlib.pyplot as plt
import random
import math
import copy
import cv2
import numpy as np

class Node(object):
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class RRT(object):
    """
    Class for RRT Planning
    """

    def __init__(self, img, start, goal, target_name):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:random sampling Area [min,max]

        """
        self.map = img
        self.start = Node(start[0][0], start[0][1])
        self.end = Node(goal[0], goal[1])
        self.expandDis = 15 # 7 need to design carefully
        self.goalSampleRate = 0.25 #0.05 
        self.maxIter = 500
        self.nodeList = [self.start]
        self.target_name = target_name

    def random_node(self):
        node_x = random.randint(170, 500)
        node_y = random.randint(160, 390)
        node = [node_x, node_y]

        return node

    @staticmethod
    def get_nearest_list_index(node_list, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index


    def planning(self):
        """
        Path planning
        """

        while True:
            # Random Sampling
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = [self.end.x, self.end.y]

            # Find nearest node
            min_index = self.get_nearest_list_index(self.nodeList, rnd)
            # print(min_index)

            # expand tree
            nearest_node = self.nodeList[min_index]

          
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)

            new_node = copy.deepcopy(nearest_node)
            new_node.x += self.expandDis * math.cos(theta)
            new_node.y += self.expandDis * math.sin(theta)
            new_node.parent = min_index
            new_node.x = int(new_node.x)
            new_node.y = int(new_node.y)
            
            lb = min(new_node.x, nearest_node.x) - 2
            rb = max(new_node.x, nearest_node.x) + 2
            tb = min(new_node.y, nearest_node.y) - 2
            db = max(new_node.y, nearest_node.y) + 2

            collision = False
            for i in range(lb, rb):
                for j in range(tb, db):
                    if not (self.map[j, i][0] == 255 & self.map[j, i][1] == 255 & self.map[j, i][2] == 255):
                        collision=True
                        break
            if collision:
                continue

            self.nodeList.append(new_node)

            # check goal
            dx = new_node.x - self.end.x
            dy = new_node.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                print("Goal!!")
                break


        path = [[self.end.x, self.end.y]]
        last_index = len(self.nodeList) - 1
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])

        return path

    def draw_static(self, path):

        rst = self.map
        for node in self.nodeList:
            if node.parent is not None:
                cv2.line(rst, (node.x, node.y), (self.nodeList[node.parent].x, self.nodeList[node.parent].y), (0,255,0), 2)

        cv2.circle(rst, (self.start.x,  self.start.y), 5, (0, 0, 255), 2)
        cv2.circle(rst, (self.end.x,  self.end.y), 5, (153, 51, 240), 2)
        
        for i in range(len(path)-1):
            cv2.line(rst, (path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]), (0, 0, 255), 2)
        cv2.imshow('result', rst)
        cv2.imwrite(self.target_name+'_path.png', rst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(img, start, goal, target_name):
    print("start RRT path planning")
    # Set Initial parameters
    rrt = RRT(img, start, goal, target_name)
    path = rrt.planning()
    print(path)
    np.save(target_name+'_path.npy', path)
    # Draw final path
    rrt.draw_static(path)


if __name__ == '__main__':
    
    starting_point = []
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            starting_point.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(img, (x, y), 7, (0, 0, 255), -1)
            cv2.imshow('image', img)
            
    targets = {"refrigerator":[(255, 0, 0), (250, 247)], "rack":[(0, 255, 133), (368, 164)], "cushion":[(255, 9, 92), (497, 266)], "lamp":[(160, 150, 20), (443, 363)], "cooktop":[(7, 255, 224),(215,314)]} 
    target = input('Enter target: ')
    print(targets[target][1])

    img = cv2.imread('map.png', 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    img = cv2.imread('map.png', 1)
    main(img, starting_point, targets[target][1], target)
