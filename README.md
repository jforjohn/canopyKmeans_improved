# canopyKmeans_improved
This is an implementation of the paper on "Improved K-means algorithm based on density Canopy".
The repo comes with a
  * *requirements.txt* for downloading possible dependencies (*pip install -r requirements.txt*)
  * *kmeans.cfg* configuration file in which you can define the specs of the algorithm you want to run

When you define what you want to run in the configuration file you just run the MainLauncher.py file.

NOTE: Don't worry about some Warnings that you may get in runtime.

Concerning the configuration file in the *data* part:
  * datasdir: the directory which contains the datasets
  * dataset: the name of the dataset with the .arff or .csv extension, which is in the same directory as this file
  * algorithm: the algorithm or the algorithms you want to run separated by a dash (-) with no spaces e.g 1-2-3. Each algorithm corresponds to a number
    * 1: Density Canopy-Kmeans (MyCanopyKMeans) [from the aforementioned paper]
    * 2: StandardCanopyKMeans
    * 3: MyKMeans
    * 4: SklearnKMeans
    * 5: MyKMeans++
    * 6: SklearnKMeans++
    * 7: MyKMedoids
    * 8: MyFuzzyCMeans
  
Concerning the configuration file in the *clustering* part:
  * normalized: boolean, to define if preprocessing should normalize the data or not (true/false)
  * canopyT1: a float, indicating the T1 parameter of the standard canopy algorithm
  * canopyT2: a float, indicating the T2 parameter of the standard canopy algorithm
  * remove_outliers: boolean, to define if density canopy should remove the outliers or not (true/false)
  * k: the number of clusters
  * tol: the tolerance for the convergence
  * max_rep: the number of maximum repetitions
  * kmeans_init_type: the type of initializing the centroids. The possible values are:
    * random: for getting random numbers following the uniform distribution
    * kmeans++: for applying KMeans++ algorithm for the initial centroids
    * canopy: for specifying that there are some centroids defined
  * run: the way you want to run the algorithms. The possible values are:
    * algorithms: for getting the indexes values for a specific k
    * silhouette: calculating the silhouette coefficient fir 15 different k and then it plots also the graph of best-k
