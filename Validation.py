
##
import matplotlib.pyplot as plt

##
def validation_metrics(df, y_true, y_pred, k_max= 15):
    from sklearn.metrics import davies_bouldin_score
    from sklearn.metrics import silhouette_score

    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import jaccard_similarity_score

    DB= davies_bouldin_score(df, y_pred)
    SC= silhouette_score(df, y_pred)
    AMI= adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ARI= adjusted_rand_score(y_true, y_pred)
    JC = jaccard_similarity_score(y_true, y_pred, normalize=True)

    return {'DB': DB, 'SC': SC, 'AMI': AMI, 'ARI': ARI, 'JC': JC}

##
#metric= validation_metrics(X, y_true, y_pred=pred)


##

def best_k(df, algo, config_file, k_max = 15):
    """
    Models:
            '1': 'KMeans',
            '2': 'KMeans++',
            '3': 'KMedoids',
            '4': 'FuzzyCMeans',
            '5': 'AggloSingle',
            '6': 'AggloAverage',
            '7': 'AggloComplete'

    """
    from sklearn.cluster import KMeans
    from config_loader import load, clf_names
    from sklearn.metrics import silhouette_score
    from MyKmeans import MyKmeans
    from MyKmedoids import MyKmedoids
    from MyFuzzyCmeans import MyFuzzyCmeans
    from sklearn.cluster import AgglomerativeClustering

    config = load(config_file)
    tol = float(config.get('clustering', 'tol'))
    max_rep = int(config.get('clustering', 'max_rep'))
    fuzzy_m = int(config.get('clustering', 'fuzzy_m'))
    kmeans_init_type = config.get('clustering', 'kmeans_init_type')
    x = [1]
    sil = [0]
    for k in range(2, k_max + 1):
        clf_options = {
            '1': MyKmeans(k, tol, max_rep),
            #'1': KMeans(n_clusters=k),
            '2': MyKmeans(k, tol, max_rep, kmeans_init_type),
            '3': MyKmedoids(k, tol, max_rep),
            '4': MyFuzzyCmeans(k, tol, max_rep, fuzzy_m),
            '5': AgglomerativeClustering(n_clusters=k, linkage='single'),
            '6': AgglomerativeClustering(n_clusters=k, linkage='average'),
            '7': AgglomerativeClustering(n_clusters=k, linkage='complete')
        }

        clf = clf_options.get(str(algo))
        clf.fit(df)
        pred = clf.labels_
        x += [k]
        sil += [silhouette_score(df, pred, metric='euclidean')]

    clf_name = clf_names.get(str(algo))
    plt.figure()
    plt.plot(x, sil, color='green', marker='o')
    plt.title('Silhouette Score ' + str(clf_name))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.ylim((0, 1))
    plt.xlim((1, k_max+1))

    return plt
##

#best_k(2)