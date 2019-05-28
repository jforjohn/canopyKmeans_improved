from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from functools import reduce
import numpy as np

class MyCanopy(BaseEstimator, TransformerMixin):
  def calc_meanDist(self, data, dists=None):
    n = data.shape[0]
    print(n)
    if dists is None:
      dists = pairwise_distances(data, metric='euclidean')
    
    if dists.shape[0] == dists.shape[1]:
      triang_dists = dists[np.arange(dists.shape[0])[:,None] > np.arange(dists.shape[1])].sum()
    else:
      ####### here!
      triang_dists = dists.sum()
    meanDist = 2*triang_dists/(n*(n-1))
    return dists, meanDist

  def p_density(self, dp, dists_dp, meanDist):
    f = np.where((dists_dp - meanDist)<0, 1, 0)
    p_i = f.sum()
    return p_i

  def a_density(self, meanDist, dists_i, p_i):
    cluster_dists = dists_i[dists_i < meanDist]
    d = cluster_dists.sum()
    return 2*d/(p_i*(p_i-1))

  def s_distance(self, p, p_i, dists_i):
    dist_i_less_j = dists_i[p-p_i>0]
    if dist_i_less_j.size > 0:
      return dist_i_less_j.min()
    else:
      return dists_i.max()
      
  def w_weight(self, p_i, a_i, s_i):
    return p_i*s_i/a_i

  def removeData(self, meanDist, dists, data, ind, centroids_dists=np.array([])):
    # in a row of dists var we have all the distances of a data point to the rest
    dists_dp = dists[ind, :]
    dist_filter = dists_dp>=meanDist
    new_dists = dists[dist_filter, :]
    if dists.shape[0] == dists.shape[1]:
      new_dists = new_dists[:, dist_filter]
    new_data = data[dist_filter, :]

    # dists from centroids
    new_centroids_dists = []
    for ind in range(centroids_dists.shape[0]):
      centroid_dists = centroids_dists[ind]
      new_centroids_dists.append(centroid_dists[dist_filter])
    new_centroids_dists = np.array(new_centroids_dists)
    return new_dists, new_data, new_centroids_dists

  def fit(self, dt):
    if isinstance(dt, pd.DataFrame):
      data = dt.values
    elif isinstance(dt, np.ndarray):
      data = dt
    else:
      raise Exception('dt should be a DataFrame or a numpy array')
    self.centroids = {}
    centroids_dists = np.array([])

    # c1
    p = np.array([[]])
    ############
    dists, meanDist = self.calc_meanDist(data)
    for ind in range(data.shape[0]):
      p = np.append(p, self.p_density(ind, dists[ind,:], meanDist))

    max_p_sample_ind = p.argmax()
    centroid = data[max_p_sample_ind, :]

    # centroid
    centroid_index = 0
    self.centroids[centroid_index] = centroid
    centroids_dists = np.concatenate([centroids_dists, dists[max_p_sample_ind, :]],
                                     axis=0).reshape(1,-1)
    print(meanDist)
    print(data.shape)
    print(dists.shape)
    dists, data, centroids_dists = self.removeData(meanDist, dists, data, max_p_sample_ind, centroids_dists=centroids_dists)
    
    print(data.shape)
    print(dists.shape)
    ############

    # c2
    p = np.array([])
    a = np.array([])
    s = np.array([])
    w = np.array([])

    ############
    _, meanDist = self.calc_meanDist(data, dists=dists)
    for ind in range(data.shape[0]):
      p_i = self.p_density(ind, dists[ind,:], meanDist)
      p = np.append(p, p_i)

      a = np.append(a, self.a_density(meanDist, dists[ind,:], p_i))
    
    for ind in range(data.shape[0]):
      s_i = self.s_distance(p, p[ind], dists[ind,:])
      s = np.append(s, s_i)
      w = np.append(w, self.w_weight(p[ind], a[ind], s_i))
    
    max_w_sample_ind = w.argmax()
    centroid = data[max_w_sample_ind, :]

    # centroid
    centroid_index += 1
    self.centroids[centroid_index] = centroid
    centroids_dists = np.concatenate([centroids_dists, [dists[max_w_sample_ind, :]]],
                                     axis=0)

    dists, data, centroids_dists = self.removeData(meanDist, dists, data, max_w_sample_ind, centroids_dists=centroids_dists)
    print(1, data.shape)
    print(2, dists.shape)
    ############
    
    p_prev = p
    s_prev = s
    while data.size > 0:
      w = np.array([])
      p = np.array([])
      a = np.array([])
      s = np.array([])

      #centroid_dists = centroid_dists.reshape(1,-1).T
      _, meanDist = self.calc_meanDist(data, dists=dists)
      for ind in range(data.shape[0]):
        p_i = self.p_density(ind, dists[ind,:], meanDist)
        p = np.append(p, p_i)

        a = np.append(a, self.a_density(meanDist, dists[ind, :], p_i))
      
      ind = 0
      while data.size > 0 and ind < data.shape[0]:
        s_i = self.s_distance(p, p[ind], dists[ind,:])
        s = np.append(s, s_i)
        # remove outliers
        if p_prev[ind] > p[ind] and s_prev[ind] < s_i:
          data = np.delete(data, ind, axis=0)
          dists = np.delete(dists, ind, axis=0)
          dists = np.delete(dists, ind, axis=1)
          p = np.delete(p, ind)
          a = np.delete(a, ind)
          p_prev = np.delete(p_prev, ind)
          s_prev = np.delete(s_prev, ind)
          centroids_dists = np.delete(centroids_dists, ind, axis=1)
          continue
 
        w_i = self.w_weight(p[ind], a[ind], s_i)
        if w.shape[0] == data.shape[0]:
          w[ind] *= w_i
        else:
          w = np.append(w, w_i)
        ind += 1
      
      max_w_sample_ind = w.argmax()
      centroid = data[max_w_sample_ind, :]
      print(centroids_dists.shape, dists.shape, data.shape)
      centroids_dists = np.concatenate([centroids_dists, [dists[max_w_sample_ind, :]]], axis=0)

      centroid_index += 1
      self.centroids[centroid_index] = centroid

      p_prev = p
      s_prev = s
      dists, data, centroids_dists = self.removeData(meanDist, dists, data, max_w_sample_ind, centroids_dists=centroids_dists)

    print(self.centroids)
        


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets

style.use('ggplot')
data = np.array([[3,5],
                 [1,4],
                 [10,12],
                 [11,13],
                 [12,10],
                 [2,3]])

iris = datasets.load_iris()
data = iris.data[:, :4]  # we only take the first two features.
y = iris.target
#plt.scatter(data[:,0], data[:,1], s=100)
#plt.show()
df = pd.DataFrame(data)
from scipy.spatial.distance import pdist
canopy = MyCanopy()
canopy.fit(data)
'''
color = ['g','c','y']
print(clf.clusters)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color=color[centroid])
    plt.scatter(data[clf.clusters[centroid][0],0], data[clf.clusters[centroid][0],1], marker='+', color=color[centroid])
plt.show()

#clf.predict(pd.DataFrame([[1,5]]))
'''