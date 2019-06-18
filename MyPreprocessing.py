##
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class MyPreprocessing:
    def __init__(self, normalized=True):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        self.normalized = normalized
        print('Normalized', normalized)

    def fit(self, data):
        df = pd.DataFrame(data)
        df = df.replace(b'?', np.NaN)

        # get label
        labels = df.iloc[:, -1]
        self.labels_ = pd.factorize(labels)[0]
        df = df.drop(df.columns[len(df.columns) - 1], axis=1)
        nan_cols = df.loc[:, df.isna().any()].columns

        # normalize numerical data
        df_num = df.select_dtypes(exclude='object')
        if df_num.size > 0:
            if self.normalized:
                min_max_scaler = preprocessing.StandardScaler()
                scaled = min_max_scaler.fit_transform(df_num.values.astype(float))
                df_normalized = pd.DataFrame(scaled, columns=df_num.columns)
            else:
                df_normalized = df_num
        else:
            df_normalized = pd.DataFrame()

        df_obj = df.select_dtypes(include='object')
        #le = preprocessing.LabelEncoder()
        #encoded = le.fit_transform(new_df)
        if df_obj.size > 0:
            df_encoded = df_obj.apply(lambda x: pd.factorize(x)[0])
            # NaN values in categorical columns are 0
            if nan_cols.size > 0:
                df_encoded.loc[:, nan_cols] += 1

            new_df_numvalues = df_encoded.values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled = min_max_scaler.fit_transform(new_df_numvalues)
            df_encoded = pd.DataFrame(scaled, columns=df_encoded.columns)
            #df_encoded = df_encoded.astype('object')
            df_encoded = df_encoded.astype('float')
        else:
            df_encoded = pd.DataFrame()

        self.new_df = pd.concat([df_normalized, df_encoded], axis=1, sort=False)

        #self.new_df = new_df.drop(new_df.columns[len(self.df.columns)-1], axis=1)
#
#print(df.select_dtypes(exclude='object'))
#print(df.select_dtypes(include='object'))
#plt.interactive(False)
#plt.show(block=True)


##
#print(agg_clustering(df_preprocess, 'Single', 3))
#agg = AgglomerativeClustering(n_clusters=2, linkage='complete')
#print(agg.fit_predict(df_preprocess))
#data, meta = loadarff('datasets/adult-test.arff')
#preprocess = MyPreprocessing(data)
#preprocess.fit()
#print(preprocess.new_df)