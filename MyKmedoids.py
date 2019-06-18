from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from itertools import permutations
from time import time
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt



class MyKmedoids:
    def __init__(self, k=2, tol=0.01, max_rep=100):
        self.k = k
        self.tol = tol
        self.max_rep = max_rep
        self.name = 'KMedoids'

    def init_medoids(self, data, init_type):
        if init_type == 'random':
            # np.random.seed(randint(1,42))
            # seeds = np.random.randint(0, len(df), self.k)
            if self.k < len(data):
                seeds = np.random.choice(len(data), self.k, replace=False)
                return seeds
            else:
                raise Exception('# of desired clusters should be < total data points')

    def find_mindist(self, data, seed):
        # print(self.medoids[seed])
        # seed_df = pd.DataFrame([self.medoids[seed]]*len(df.index))
        return distance_metric(data, self.medoids[seed])

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        else:
            raise Exception('dt should be a DataFrame or a numpy array')

        # get random indexes from data
        self.seeds = self.init_medoids(data, 'random')
        self.clusters = {}
        self.medoids = {}

        for seed in range(self.k):
            self.medoids[seed] = data[seed, :]

        print('Calculate distance matrix')
        start = time()
        #distances = distance_matrix(data, data, p=2)
        distances = pairwise_distances(data) #, data, p=2)
        print('Duration for distance matrix: %s' %(time()-start))

        converge = False
        while self.max_rep > 0 and converge == False:
            print(self.max_rep)
            dist2medoids = np.array([distances[seed, :] for seed in self.seeds])
            #dist2medoids = np.array([get_dp_distances(distances, seed) for seed in self.seeds])
            # dist2medoids has k columns which correspond to the dist from each medoid
            # dist_df = pd.concat(dist2medoids, axis=1).idxmin(axis=1)
            self.labels_ = dist2medoids.argmin(axis=0)

            for seed_index in range(self.k):
                self.clusters[seed_index] = np.where(
                    self.labels_ == seed_index)[0]

            prev_medoids = self.medoids.copy()
            #prev_seeds = self.seeds
            self.medoids = {}
            new_seeds = []
            for seed_index in range(self.k):
                #clusters_sse = []
                min_dp_cluster = None
                medoid = None
                cluster = self.clusters[seed_index]
                for dp in cluster:
                    #dp_cluster_dist = sum(get_dp_distances(distances, dp, cluster))
                    dp_cluster_dist = distances[dp, cluster].sum()
                    # initial value of min_dp_cluster
                    if not min_dp_cluster:
                        min_dp_cluster = dp_cluster_dist
                        medoid = dp

                    if min_dp_cluster > dp_cluster_dist:
                        min_dp_cluster = dp_cluster_dist
                        medoid = dp
                '''
                clusters_sse.append(dp_cluster_dists)
                
                for dp1, dp2 in permutations(self.clusters[seed_index], 2):
                    if dp1 == dp1_prev:
                        sum += calc_distances(distances, dp1, dp2)
                    else:
                        cluster_sse.append(sum)
                        sum = 0
                        dp1_prev = dp1
                        sum += calc_distances(distances, dp1, dp2)
                
                #clusters_sse.append(sum)
                new_cluster_seed = np.array(cluster_sse).argmin(axis=0)
                
                new_seeds.append(
                    self.clusters[seed_index][new_cluster_seed])
                '''
                new_seeds.append(medoid)
                self.medoids[seed_index] = data[medoid]
            self.seeds = np.array(new_seeds)
            #self.medoids = {seed_index:data[self.seeds[seed_index]]
            #               for seed_index in range(self.k)}

            #self.centroids[seed] = data[self.clusters[seed]].mean(axis=0)

            converge = True
            for seed_index in range(self.k):
                #if euclidean(prev_medoids[seed_index], self.medoids[seed_index]) <= self.tol:
                dist_diff = np.linalg.norm(prev_medoids[seed_index]-self.medoids[seed_index],
                                           ord=2)
                if dist_diff <= self.tol:
                    converge = converge and True
                else:
                    converge = converge and False

            self.max_rep -= 1

        print('Remaining repetitions: %s' %(self.max_rep))

        self.inertia_ = 0
        for seed in self.medoids:
            self.inertia_ += np.array([self.find_mindist(data[self.clusters[seed]], seed)**2]).sum()
    '''
    def predict(self, df):
        dist2medoids = [self.find_mindist(df, seed) for seed in self.medoids]
        # dist2medoids has k columns which correspond to the dist from each centroid
        dist_df = pd.concat(dist2medoids, axis=1).idxmin(axis=1)

        for seed in self.medoids:
            if (dist_df == seed)[0]:
                return seed
    '''
def get_dp_distances(distances, row_idx, *c_members_tup):

    if len(c_members_tup) == 0:
        '''
        keys = np.array(list(distances.keys()))
        x = np.where(
                keys[:,0] == row_idx
            )[0]
        y = np.where(
                keys[:,1] == row_idx
            )[0]
        dp = keys[np.append(x,y),:]

        '''
        dp_dists = map(lambda x: distances[x]
                       if x[0]==row_idx or x[1]==row_idx else None,
                       distances)
        dp_dists = list(filter(None.__ne__, dp_dists))

    else:

        cluster_members =  c_members_tup[0]

        dp_dists = map(lambda x: distances[x]
            if ((x[0] == row_idx and x[1] in cluster_members) or
               (x[1] == row_idx and x[0] in cluster_members)) else None,
            distances)
        dp_dists = list(filter(None, dp_dists))
    return dp_dists

def custom_distance_matrix(data):
    rows, _ = data.shape
    distances = {} #np.zeros([rows, rows]) #- 1
    for rowid in range(rows):
        for colid in range(rowid+1):
            distances[(rowid, colid)] = np.linalg.norm(
                data[rowid, :] - data[colid, :])
    ## copy lower diagonal values to the upper side
    #index_upper = np.triu_indices(rows)
    #distances[index_upper] = distances.T[index_upper]
    return distances

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
        # a_num = a.select_dtypes(exclude='object')
        # a_cat = a.select_dtypes(include='object')
        ## make the same size as a
        # b_num = b.select_dtypes(exclude='object')
        # b_cat = b.select_dtypes(include='object')
        # print(a)
        # print(a-b)
        distance = ((a - b) ** 2).sum(axis=1)

        # dist_cat = pd.DataFrame(np.where(a_cat==b_cat, 0, 1)).sum(axis=1)
        # return (distance + dist_cat)**0.5
        return distance ** 0.5

'''
data = np.array([[2,3],
                 [3,5],
                 [1,4],
                 [10,12],
                 [11,13],
                 [12,10]])

plt.scatter(data[:,0], data[:,1], s=100)
#lt.show()
df = pd.DataFrame(data)

clf = MyKmedoids(k=2)
clf.fit(df)
print(clf.clusters)
print(clf.medoids)


color = ['g','y','c']
print(clf.clusters)
for centroid in clf.medoids:
    plt.scatter(clf.medoids[centroid][0], clf.medoids[centroid][1], marker='o', color=color[centroid])
    plt.scatter(data[clf.clusters[centroid],0], data[clf.clusters[centroid],1], marker='+', color=color[centroid])
plt.show()
'''

#clf.predict(pd.DataFrame([[1,5]]))