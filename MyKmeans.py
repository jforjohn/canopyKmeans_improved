from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from random import randint
from MyCanopy import MyCanopy
style.use('ggplot')

class MyKmeans:
    def __init__(self, k=2, tol=0.01, max_rep=100,
                 init_type='random', init_centers=None):
        self.k = k
        self.tol = tol
        self.max_rep = max_rep
        self.centroids = {}
        self.clusters = {}
        self.init_type = init_type
        if init_type == 'random':
            self.name = 'KMeans'
        elif init_type == 'kmeans++':
            self.name = 'KMeans++'
        elif init_type == 'canopy':
            self.centroids = init_centers
            self.k = len(init_centers)
            self.name = 'CanopyKmeans'
        else:
            self.name = 'NotSpecified'

    def init_centroids(self, data, init_type):
        if self.k < len(data):
            if init_type == 'random':
            #np.random.seed(randint(1,42))
            #seeds = np.random.randint(0, len(df), self.k)
                seeds = np.random.choice(len(data), self.k, replace=False)
                for index in range(self.k):
                    self.centroids[index] = data[seeds[index], :]

            elif init_type == 'kmeans++':
                len_data = data.shape[0]
                seed1 = np.random.choice(len_data, 1, replace=False)[0]
                centroid_index = 0
                self.centroids[centroid_index] = data[seed1, :]
                seeds = [seed1]

                # Here starts the for-loop for the other seeds:
                for cluster_index in range(self.k - 1):
                    dist2centroids = np.array([self.find_mindist(data, seed)
                                               for seed in self.centroids])**2
                    #dist_df = dist2centroids.argmin(axis=0)
                    for seed in seeds:
                        dist2centroids[:, seed] = 0
                    dist_sum = dist2centroids.sum()
                    D2 =  (dist2centroids/dist_sum).sum(axis=0)

                    #cumprobs = D2.cumsum()
                    #r = np.random.uniform(0, 1)
                    new_seed = np.random.choice(len_data, 1, replace=False, p=D2)[0]
                    seeds.append(new_seed)
                    centroid_index += 1
                    self.centroids[centroid_index] = data[new_seed, :]
            '''
            elif init_type == 'canopy':
                canopy = MyCanopy()
                canopy.fit(data)
                self.centroids = canopy.centroids
                self.k = len(self.centroids)
            '''

        else:
            raise Exception('# of desired clusters should be < total data points')


    def find_mindist(self, data, seed):
        #print(self.centroids[seed])
        #seed_df = pd.DataFrame([self.centroids[seed]]*len(df.index))
        return distance_metric(data, self.centroids[seed])

    def handle_empty_cluster(self, dist2centroids, data, seed, emptySeeds):
        #choose non empty seeds from distance matrix
        nonEmpty_dist2centroids = np.delete(dist2centroids, emptySeeds, axis=0)
        dat_point_maxDist = nonEmpty_dist2centroids.sum(axis=0).argmax()
        self.centroids[seed] = data[dat_point_maxDist, :]
        return np.array(self.find_mindist(data, seed))

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        else:
            raise Exception('dt should be a DataFrame or a numpy array')

        if not len(self.centroids) == self.k:
            print('No init centers', self.name)
            # get random indexes from data
            self.init_centroids(data, self.init_type)

        converge = False
        while self.max_rep > 0 and converge == False:
            emptyCluster = False
            emptySeeds = []
            dist2centroids = np.array([self.find_mindist(data, seed)
                              for seed in self.centroids])
            # dist2centroids has k rows which correspond to the dist from each centroid
            #dist_df = pd.concat(dist2centroids, axis=1).idxmin(axis=1)
            self.labels_ = dist2centroids.argmin(axis=0)

            for seed in self.centroids:
                self.clusters[seed] = np.where(self.labels_==seed)
                if self.clusters[seed][0].size == 0:
                    print("Cluster %s with centroid %s is empty!"
                          %(seed, self.centroids[seed]))
                    emptySeeds.append(seed)
                    emptyCluster = True

            # check for empty clusters
            if emptyCluster:
                for seed in emptySeeds:
                    dist2centroids[seed] = self.handle_empty_cluster(dist2centroids, data, seed, emptySeeds)
                    emptySeeds.pop(emptySeeds.index(seed))

                # find new clusters after fixing empty ones
                self.labels_ = dist2centroids.argmin(axis=0)
                for seed in self.centroids:
                    self.clusters[seed] = np.where(self.labels_ == seed)


            prev_centroids = self.centroids.copy()
            for seed in self.clusters:
                self.centroids[seed] = data[self.clusters[seed]].mean(axis=0)

            converge = True
            for seed in self.clusters:
                #if euclidean(prev_centroids[seed], self.centroids[seed]) <= self.tol:
                #if np.array_equal(prev_centroids[seed], self.centroids[seed]):
                dist_diff = np.linalg.norm(prev_centroids[seed]-self.centroids[seed],
                                      ord=2)
                if dist_diff < self.tol:
                    converge = converge and True
                else:
                    converge = converge and False

            self.max_rep -= 1
        print('Remaining repetitions: %s' % (self.max_rep))
        
        self.inertia_ = 0
        for seed in self.centroids:
            self.inertia_ += np.array([self.find_mindist(data[self.clusters[seed]], seed)**2]).sum()

def distance_metric(a, b, dist='Euclidean'):
    """
    Define the distance metric used
    This can be: 'Euclidean' (default)
    """
    # a numpy matrix, b numpy vector of the centroid
    if a.shape[1] == b.shape[0]:
        """
        We assume that:
        - the numerical values of a and are normalized
        - a and b have the same columns from now on
        """
        #a_num = a.select_dtypes(exclude='object')
        #a_cat = a.select_dtypes(include='object')
        ## make the same size as a
        #b_num = b.select_dtypes(exclude='object')
        #b_cat = b.select_dtypes(include='object')
        #print(a)
        #print(a-b)
        distance = ((a - b)**2).sum(axis=1)

        #dist_cat = pd.DataFrame(np.where(a_cat==b_cat, 0, 1)).sum(axis=1)
        #return (distance + dist_cat)**0.5
        return distance**0.5
##
'''
data = np.array([[2,3],
                 [3,5],
                 [1,4],
                 [10,12],
                 [11,13],
                 [12,10]])

plt.scatter(data[:,0], data[:,1], s=100)
#plt.show()
df = pd.DataFrame(data)

clf = MyKmeans(k=3, init_type='kmeans++')
clf.fit(df)
color = ['g','c','y']
print(clf.clusters)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color=color[centroid])
    plt.scatter(data[clf.clusters[centroid][0],0], data[clf.clusters[centroid][0],1], marker='+', color=color[centroid])
plt.show()

#clf.predict(pd.DataFrame([[1,5]]))
'''
