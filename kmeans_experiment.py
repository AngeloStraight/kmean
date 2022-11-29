from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from decimal import Decimal
import numpy as np
import copy
import math
import heapq
import time


class KMeansExperiment:
    def __init__(self, *args,**kwargs):
        self.data = args
        self.r_data = kwargs
        self.X = args[0] if args else None 
        self.first_x = None
        self.vor = None
        self.kmeans = None
        self.k = None
        self.furthest_point = None
        self.f_points = []
        self.closest_center = None
        self.poison = []
        # self.bbox = None
        self.poison_count_arr = np.array([])
        self.poison_vs_objective = []
        self.scores = np.array([])
        self.max_dist = float('-inf')
        self.size_to_poision_proportion = None
        self.max_location = np.array([float('-inf'),float('-inf')], dtype=object)
        self.is_random = True if 'random' in self.r_data.keys() else False
        self.__compute()


    def __compute(self):
        if self.is_random:
            if not ('low' in self.r_data.keys() and 'high' in self.r_data.keys() and 'size' in self.r_data.keys()):
                print("Pass low, high, and size as keyword arguments")
                exit()

            """ set bounding box """
            if not ('p1' in self.r_data.keys() and 'p2' in self.r_data.keys()):
                print("Need to include two points p1, p2 for bounding box")
                exit()

            # self.bbox = BoundingBox([self.r_data['p1'],self.r_data['p2']])

            """ set random data """
            self.X = np.random.uniform(low=self.r_data['low'], high=self.r_data['high'], size=self.r_data['size'])
            self.first_x = copy.deepcopy(self.X)
            
        else:
            self.first_x = copy.deepcopy(self.X)
            min_point = np.array([float('inf'),float('inf')], dtype=object)
            max_point = np.array([float('-inf'),float('-inf')], dtype=object)
            for p in self.X:
                if p[0] <= min_point[0] and p[1] <= min_point[1]:
                    min_point = p

                if p[0] >= max_point[0] and p[1] >= max_point[1]:
                    max_point = p

            max_point = [round(num) for num in max_point]
            min_point = [round(num) for num in min_point]
            # self.bbox = BoundingBox([(min_point[0], min_point[1]),(max_point[0], max_point[1])])


        """ Check if the number of clusters were provided """
        if 'k' in self.r_data.keys():
            self.k = self.r_data['k']

        """ Initiate kmeans clustering classification algorithm """
        self.kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.X)

        """ Create a voronoi partition of the data """
        # self.vor = Voronoi(self.kmeans.cluster_centers_)
        self.vor = Voronoi(self.X)

        

        self.__compute_furthest_point()


    # def get_bbox(self):
    #     return self.bbox

    def get_furthest_point(self):
        return self.furthest_point

    def get_original_data(self):
        return self.first_x

    def get_poisoned_data(self):
        return self.X
    
    def get_k_centers(self):
        return self.kmeans.cluster_centers_

    def get_original_centers(self):
        original_kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.first_x)
        return original_kmeans.cluster_centers_

    def __compute_furthest_point(self):

        """ Check if furthest point is one of the voronoi vertices. """
       
        min_heap = []
        
        for p in self.vor.vertices:
            if (p[0]  >= 0 and p[0] <= 1) and ( p[1]  >= 0 and p[1] <= 1):
                temp_heap = []
                
                for c in self.kmeans.cluster_centers_:
                    heapq.heappush(temp_heap, (distance.euclidean(c, p), p))
                heapq.heappush(min_heap, heapq.heappop(temp_heap))

        self.furthest_point = heapq.nlargest(1,min_heap)


        """ Preparing the bounding box for testing """
        bottom_line = np.linspace(0, 1, num=5) # bottom line
        top_line    = np.linspace(0, 1, num=5) # top line
        left_line   = np.linspace(0, 1, num=5) # far left line
        right_line  = np.linspace(0, 1, num=5) # far left line
        
        # range of points on each line
        bl = [np.array([x,0], dtype=object) for x in bottom_line]
        tl = [np.array([x,1], dtype=object) for x in top_line]
        ll = [np.array([0, x], dtype=object) for x in left_line]
        rl = [np.array([1, x], dtype=object) for x in right_line]

        box_points = np.concatenate((bl, tl), axis=0)
        box_points = np.concatenate((box_points, ll), axis=0)
        box_points = np.concatenate((box_points, rl), axis=0)
        # print(box_points)
        x1 = np.random.rand(box_points.shape[1])
        # print(x1)

        y = box_points.dot(x1)
        unique, index = np.unique(y, return_index=True)
        box_points[index]
        min_heap.clear()

        """ Check if furthest point is on bounding box instead of voronoi vertice """
        for p in box_points[index]:
            temp_heap = []
            
            for c in self.kmeans.cluster_centers_:
                if list(p) not in [list(x) for x in self.f_points]:
                    heapq.heappush(temp_heap, (distance.euclidean(c, p), p))
            if temp_heap:
                heapq.heappush(min_heap, heapq.heappop(temp_heap))

        if self.furthest_point[0][0] < heapq.nlargest(1,min_heap)[0][0]:
            self.furthest_point = heapq.nlargest(1,min_heap)[0][1]
        else:
            self.furthest_point = self.furthest_point[0][1]

        
    def show_plot(self):
        red = [1, 0, 0]
        blue = [0, 0, 1]
        colors = [red]
        x = self.kmeans.cluster_centers_[:,0]
        y = self.kmeans.cluster_centers_[:,1]

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(self.X[:,0], self.X[:,1], c=[blue])
        ax.scatter(x, y, s=300, marker='*', c=colors)

        plt.show()

    def find_closest_center(self):
        self.closest_center = np.array([float('inf'),float('inf')], dtype=object)
        min_distance = float('inf')

        for c in self.kmeans.cluster_centers_:
            temp_dist = distance.euclidean(self.furthest_point, c)
            if temp_dist < min_distance:
                self.closest_center = c
                min_distance = temp_dist

    def lin_equ(self, l1, l2):
        """Line encoded as l=(x,y)."""
        m = Decimal((l2[1] - l1[1])) / Decimal(l2[0] - l1[0])
        c = (Decimal(l2[1]) - (m * Decimal(l2[0])))
        return m, c

        # # Example Usage:
        # lin_equ((-40, 30,), (20, 45))

    def __get_point(self, x, m, c):
        # y = mx + b
        return (float(x),float(m*Decimal(x) + c))

# Result: (Decimal('0.25'), Decimal('40.00'))
   
    def show_analysis(self, name):
        blue = [0, 0, 1]

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(self.poison_count_arr, self.scores, c=[blue])

        plt.title(name)
        plt.xlabel("Number of Poison Points (m)")
        plt.ylabel("Objective Function Score")

        plt.show()

    
    def add_poison_line(self, m, step=10):
        
        # if len(self.poison) == 0:
        # compute closest center and set interval for line
        self.find_closest_center()

        poison_list = []
        # start = time.time()
        # while total_poison < m:

        lower_x = self.closest_center[0] if self.closest_center[0] <= self.furthest_point[0] else self.furthest_point[0]
        upper_x = self.closest_center[0] if self.closest_center[0] >= self.furthest_point[0] else self.furthest_point[0]
    
        # find the length of the line, partitions, and how much poison for each partition
        line_magnitude = upper_x - lower_x
        delta = line_magnitude / (self.k+1)

        alpha = math.ceil(m/(self.k+1))

        # use line equation function to get slope and y-intercept
        eq = self.lin_equ(self.closest_center, self.furthest_point)

        # poison the line with proper amount of poison for each partition 
        poison = []
        while lower_x <= (upper_x):
            for x in np.linspace(lower_x, lower_x + delta, num=alpha):
                if x > upper_x:
                    break
                poison.append(list(self.__get_point(x, eq[0], eq[1])))
            lower_x += delta
        poison = np.array(poison)

        # add poison to 
        
        self.X = np.append(self.X, poison, axis=0)
        poison_list.append(poison)
        
        self.poison = np.array([p for sub in poison_list for p in sub])


    def add_t_furthest_points(self, t):
        self.f_points.append(self.furthest_point)             # we already computed the first furthest point in _compute() call
        for i in range(1, t):
            self.X = np.append(self.X, np.array([self.furthest_point]), axis=0)
            self.kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.X)
            self.vor = Voronoi(self.X)
            self.__compute_furthest_point()
            self.f_points.append(self.furthest_point)
        self.f_points = np.array(self.f_points)

    def get_f_points(self):
        return self.f_points





    def add_poison_point(self, m, step=10, split=False):
        

        """ Create a voronoi partition of the data """
        # self.vor = Voronoi(self.kmeans.cluster_centers_)
        # self.vor = Voronoi(self.X)
        # self.__compute_furthest_point()
       
        if len(self.poison) == 0:
            total_poison = copy.deepcopy(step)
            start = time.time()
            while total_poison < m:
                if split:
                    self.poison = np.array(self.f_points * (total_poison / self.f_points.shape[0]))
                else:
                    self.poison = np.array([self.furthest_point]*total_poison)

                self.X = np.append(self.X, self.poison, axis=0)
                self.kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.X)
                self.poison_vs_objective.append([total_poison, -1 * self.kmeans.score(self.X)])
                if time.time() - start > 60:
                    total_poison *= 2
                    start = time.time()
                else:
                    total_poison += step
                self.X = copy.deepcopy(self.first_x)

    def get_poison_vs_objective(self):
        return self.poison_vs_objective

    def get_poison(self):
        return self.poison

    def get_score(self):
        return -1 * self.kmeans.score(self.X)

    def run_analysis(self, m, line=True, step=10):

        total = step
        if line:
            start = time.time()
            while total < m: 
                self.add_poison_line(total, step)
                self.kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.X)
                self.poison_vs_objective.append([total, -1 * self.kmeans.score(self.X)])
                if time.time() - start > 60:
                    total *= 2
                    start = time.time()
                else:
                    total += step
                
                self.X = copy.deepcopy(self.first_x)
            

    def get_analysis(self):
        return self.poison_count_arr, self.scores

    def show_vor(self):
        if self.vor:
            fig = voronoi_plot_2d(self.vor)
            plt.show()
        else:
            print("need to set voronoi")

    def get_poison_location(self):
        return self.max_location


    def show_poison_line(self, split=False):

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(self.X[:,0], self.X[:,1], c=[[0, 0, 0]]) # original cluster
        
        ax.scatter( self.kmeans.cluster_centers_[:,0], self.kmeans.cluster_centers_[:,1], s=500, marker='*', color='green')             # k mean centers
        plt.text(self.poison[:,0][-1], self.poison[:,1][-1]+ 0.05,f"|m| = {len(self.poison)}", size=20)
        if not all(x==self.poison[:,0][0] for x in self.poison[:,0]) and not split:
            ax.scatter(self.poison[:,0], self.poison[:,1], color='red')             # m poision points (Line)
            # ax.scatter(self.poison[:,0: len(self.poison[:,0])//3]  , self.poison[:,1: len(self.poison[:,1])//3], color='red')
            ax.scatter(self.furthest_point[0], self.furthest_point[1], s=500, marker='*', color='navy')
        else:
            ax.scatter(self.poison[:,0], self.poison[:,1], color='red', s=500) 
            # ax.scatter(self.poison[:,0:len(self.poison[:,0])//3], self.poison[:,1], color='red', s=500)
    
        plt.show()