
##
# Load Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_csv("data/jokes_final.csv")
data = data.fillna(99.0)

counts = data['0'].values
data = data.drop(['0'], axis=1)
data = data.drop([0])
users = [i for i in range(data.shape[0])]
# gauge = [i for i in range(1,6)]
jokes = [i for i in range(6,data.shape[1]+1)]
n = data.shape[0]
m = data.shape[1]
k = 5

gauge_cols = ['7','8','13','15','16']
setA = data[[i for i in gauge_cols]]

##
# Normalize Data


def standardize_col(col):
    mean = np.mean(col)
    std = np.std(col)
    return col.apply(lambda x: (x - mean) / std)


for col in setA.columns:
    setA[col] = standardize_col(setA[col])
##
# PCA Analysis
from sklearn.decomposition import PCA

setC = np.matmul(setA.values.T, setA.values)
setC = np.dot(setC,(1/(setA.shape[0] -1)))

covar_matrix = PCA(n_components = 5)
covar_matrix.fit(setA)
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)

import matplotlib.pyplot as plt
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')

plt.plot(var)
plt.show()

##
# Eigen Decomposition

w,setE = np.linalg.eig(setC)
setA1 = np.diagflat(w)
setP = np.matmul(setA, setE.T)


##
# Normalize Dataset

setJ = data.drop(gauge_cols, axis=1)

avg_rating = []
def standardize_col_skip_99(col):
    # rated = np.where(col <= 10.)
    arr1 = [x for x in col if x <= 10.]
    if len(arr1) == 0:
        mean = 0
    else:
        mean = np.mean(arr1)
    avg_rating.append(mean)
    std = np.std(arr1)
    return col.apply(lambda x: (x - mean) / std if x <= 10 else 99.0)


for col in setJ.columns:
    setJ[col] = standardize_col_skip_99(setJ[col])

# print(avg_rating)

##
# Clustering

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, max_iter=10, random_state=1).fit(setP)

labels = kmeans_model.labels_
cluster_map = pd.DataFrame()
cluster_map['data_index'] = [x for x in range(1, setP.shape[0]+1)]
cluster_map['cluster'] = labels

for i in np.unique(labels):
    print(len(cluster_map[cluster_map['cluster'] == i]))

##
# Metadata Computation

joke_map = pd.DataFrame(np.zeros((len(np.unique(labels)), len(setJ.columns))), columns=setJ.columns)
for i in np.unique(labels):
    cluster_temp = cluster_map[cluster_map['cluster'] == i]
    for j in tqdm(joke_map.columns):
        rating = []
        for index in cluster_temp['data_index']:
            if setJ.loc[index, j] <= 10:
                rating.append(setJ.loc[index, j])
            else:
                pass

        joke_map.loc[i, j] = np.mean(rating)


joke_map = joke_map.fillna(0)

# joke_map.to_csv('/content/drive/My Drive/CS550/jokemap.csv', index=False)
# cluster_map.to_csv('/content/drive/My Drive/CS550/cluster_map.csv', index=False)


##
# Prediction

input = [1]
for user in input:
    unrated = np.where(setJ.loc[user] > 10.)[0]
    unrated = [setJ.columns[i] for i in unrated]
    cluster = labels[user]
    joke_rec =[]
    for i in unrated:
        joke_rec.append(joke_map.iloc[cluster][str(i)])
    print(str((- np.array(joke_rec)).argsort()[:10]))

##
# Playground

plt.hist(data.loc[:, "0"].loc[data["0"] != 99.0])
plt.show()