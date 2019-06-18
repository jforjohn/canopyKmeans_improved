from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from itertools import permutations

class MyFuzzyCmeans:
    def __init__(self, k=2, tol=0.001, max_rep=100, m=2.0):
        self.k = k
        self.tol = tol
        self.max_rep = max_rep
        # Fuzzy parameter
        self.m = m
        self.name = 'FuzzyCMeans'

    def init_centers(self, data, init_type):
        if init_type == 'random':
            values =  np.random.rand(self.k, data.shape[0])
            values_sum = values.sum(axis=0)
            return values/values_sum

    def new_universe_matrix(self, universe_matrix, data):
        power_num = float(2 / (self.m - 1))
        # dists: k rows with value the distance between each data point and the i-th centre
        dists = np.array([np.linalg.norm(data-v_centre, ord=1, axis=1)
                 for v_centre in self.v_centres])
        #den = dists.sum(axis=0)
        new_u = []
        n = data.shape[0]
        for data_index in range(n):
            for cluster_index in range(self.k):

                self.universe_matrix[cluster_index, data_index] = (
                    1/sum(
                        [(dists[cluster_index, data_index]/dists[c_index, data_index])**power_num
                         for c_index in range(self.k)]
                        )
                )

    def v_centres_calc(self, data):
        universe_matrix_m = self.universe_matrix ** self.m
        v_centres_num = np.matmul(universe_matrix_m, data)
        # make the den a column vector with reshape
        v_centres_den = universe_matrix_m.sum(axis=1).reshape(-1, 1)

        self.v_centres = v_centres_num / v_centres_den

    def find_mindist(self, data, seed):
        #print(self.centroids[seed])
        #seed_df = pd.DataFrame([self.centroids[seed]]*len(df.index))
        return distance_metric(data, self.v_centres[seed])

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        else:
            raise Exception('dt should be a DataFrame or a numpy array')

        # get random indexes from data
        self.universe_matrix = self.init_centers(data, 'random')

        converge = False
        while self.max_rep > 0 and converge == False:
            if not hasattr(self, 'v_centres'):
                self.v_centres_calc(data)
            v_centres_old = self.v_centres.copy()
            self.new_universe_matrix(self.universe_matrix, data)
            self.v_centres_calc(data)

            #print(self.v_centres)
            converge = True
            for cluster_index in range(self.k):
                dist_diff = np.linalg.norm(self.v_centres[cluster_index]-v_centres_old[cluster_index],
                                      ord=1)
                if dist_diff <= self.tol:
                    converge = converge and True
                else:
                    converge = converge and False

            self.max_rep -= 1

        self.labels_ = self.universe_matrix.argmax(axis=0)
        #print(self.v_centres)
        print('Remaining repetitions: %s' % (self.max_rep))

        self.inertia_ = 0
        for seed in  range(len(self.v_centres)):
            self.inertia_ += np.array([self.find_mindist(data[np.where(self.labels_==seed)], seed)**2]).sum()

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
'''
clf = MyFuzzyCmeans()
data = np.array([[2,3],[3,4],[1,5], [10,9], [12,13], [13,14],[11,15]])
clf.fit(data)
print(clf.universe_matrix)
print(clf.labels_)
'''
