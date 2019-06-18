##
from sklearn.cluster import KMeans
from MyKmeans import MyKmeans
from MyKmedoids import MyKmedoids
from MyFuzzyCmeans import MyFuzzyCmeans
from MyPreprocessing import MyPreprocessing
from sklearn.cluster import AgglomerativeClustering
from MyCanopy import MyCanopy
from standardCanopy import myStandardCanopy
from scipy.io.arff import loadarff
import pandas as pd
from config_loader import load, clf_names
from sklearn.datasets import make_moons, make_blobs
import argparse
from os import path
import sys
from time import time
import numpy as np
import pandas as pd

##
from Validation import validation_metrics
from Validation import best_k

##
if __name__ == '__main__':
    ##

    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="kmeans.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    ##
    datadir = config.get('data', 'datadir')
    dataset = config.get('data', 'dataset')
    path = path.join(datadir, dataset)
    try:
        if dataset.split('.')[-1] == 'arff':
            data, meta = loadarff(path)
        else:
            # first row is used as header
            data = pd.read_csv(path)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" %(dataset, path))
        sys.exit(1)

    ##
    k = int(config.get('clustering', 'k'))
    tol = float(config.get('clustering', 'tol'))
    max_rep = int(config.get('clustering', 'max_rep'))
    fuzzy_m = int(config.get('clustering', 'fuzzy_m'))
    kmeans_init_type = config.get('clustering', 'kmeans_init_type')
    run = config.get('clustering', 'run')

    ## Preprocessing
    normalized = config.getboolean('clustering', 'normalized')
    preprocess = MyPreprocessing(normalized=normalized)
    preprocess.fit(data)
    df = preprocess.new_df
    labels = preprocess.labels_

    # MyCanopy
    remove_outliers = config.getboolean('clustering', 'remove_outliers')
    mycanopy = MyCanopy(remove_outliers=remove_outliers)
    mycanopy.fit(df)
    #centers = list(mycanopy.centroids.values())
    #print('Canopy centers', centers)
    
    # StandardCanopy
    t1 = config.getfloat('clustering', 'canopyT1')
    t2 = config.getfloat('clustering', 'canopyT2')
    standard_canopy = myStandardCanopy(df, t1, t2)
    centroids = {}
    for i in standard_canopy.keys():
        centroid_info = standard_canopy[i]
        centroid = df.iloc[centroid_info['c'],:]
        centroids[i] = centroid.values

    clf_options = {
        '1': MyKmeans(k, tol, max_rep, 'canopy', mycanopy.centroids),
        #'1': KMeans(n_clusters=len(centers), tol=tol, max_iter=max_rep, init=np.array(centers)),
        '2': MyKmeans(k, tol, max_rep, 'canopy', centroids),
        '3': MyKmeans(k, tol, max_rep),
        '4': KMeans(n_clusters=k, tol=tol, max_iter=max_rep, init='random'),
        '5': MyKmeans(k, tol, max_rep, 'kmeans++'),
        '6': KMeans(n_clusters=k, tol=tol, max_iter=max_rep, init='k-means++'),
        '7': MyKmedoids(k, tol, max_rep),
        '8': MyFuzzyCmeans(k, tol, max_rep, fuzzy_m),
    }

    algos = config.get('data', 'algorithm').split('-')
    values = pd.DataFrame()
    for algo in algos:
        #print(df.values)
        #print(df.dtypes)

        ##

        clf = clf_options.get(str(algo))

        clf_name = clf_names.get(str(algo))
        if not clf:
            print("Not available algorithm defined in config file. Available options:%s"
                  % (clf_options.keys()))
            sys.exit(1)
        print('Algorithm %s' % (clf_name))
        if run == 'algorithms':
            start = time()
            clf.fit(df)
            duration = time() - start
            '''
            if hasattr(clf, 'centroids'):
                print('Final centroids', clf.centroids)
            else:
                print('Final centroids', clf.cluster_centers_)
            '''
            metrics = validation_metrics(df, labels, clf.labels_)
            if hasattr(clf, 'max_rep'):
                rep = max_rep - clf.max_rep
            else:
                rep = clf.n_iter_
            if hasattr(clf, 'k'):
                k = clf.k
            else:
                k = len(clf.cluster_centers_)
            metrics.update({"ERR": clf.inertia_,
                            "TD": duration,
                            "REP": rep,
                            'K': int(k),
                            'Dataset': dataset})
            validations = pd.DataFrame.from_dict(metrics, orient='index',
                                                 columns=[clf_name])
            values = pd.concat([values, validations], axis=1)
            # print(clf.clusters)
            print()
            print('---')
            print()


        elif run == 'silhouette':
            best_k(df, algo, config_file).show()

    print(values)
    #values.to_csv(f'ad_results.csv', mode='a', header=False, index=False)