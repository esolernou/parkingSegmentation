# %% llibreries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('data/users_feb_pca.csv')
X = users_feb.iloc[:,1:]
X.info()
# %%  normalització dades
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% SSE - suma errors quadràtics
K = range(2, 20)
SSEs = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    dists = kmeans.fit_transform(X)
    SSEs.append(np.sum(np.min(dists, axis=1)**2))

plt.plot(K, SSEs, 'gx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method (X)')
plt.show()
# %%  silhouette
K = range(2, 20)
silhouettes = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    sil = metrics.silhouette_score(X, labels, sample_size=20000)
    silhouettes.append(sil)
    print('k: ', k, 'silhouette: ', sil)

plt.plot(K, silhouettes, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.title('Elbow Method (X)')
plt.show()

# %% SSE - suma errors quadràtics
K = range(2, 30)
SSEs = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    dists = kmeans.fit_transform(X_scaled)
    SSEs.append(np.sum(np.min(dists, axis=1)**2))

plt.plot(K, SSEs, 'gx-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.title('Elbow Method (X_scaled)')
plt.show()

# %%  silhouette
K = range(2, 20)
silhouettes = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouettes.append(metrics.silhouette_score(X_scaled, labels,
                                                metric='euclidean'))

plt.plot(K, silhouettes, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.title('Elbow Method (X_scaled)')
plt.show()

# %% número de clústers - average distortion
K = range(2, 30)
meanDistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(X)
    meanDistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Elbow Method for euclidean distance (X)')
plt.show()

# %%
K = range(2, 40)
meanDistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(X_scaled)
    meanDistortions.append(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Elbow Method for euclidean distance (X_scaled)')
plt.show()

#%%
# Generem model amb k=6 i dades estandaritzades
kmeans6 = KMeans(n_clusters=6,init='k-means++')
kmeans6.fit(X_scaled)
# guardem les etiquetes
X['kmeans6'] = kmeans6.labels_
centers_kmeans6 = pd.DataFrame(scaler.inverse_transform(kmeans6.cluster_centers_), columns=X.iloc[:,:-1].columns)
sil_k6 = metrics.silhouette_score(X_scaled, kmeans6.labels_, metric='euclidean')
#%% centres dels clústers
scaler.inverse_transform(kmeans6.cluster_centers_)
#%% guardem dades etiquetades
X.to_csv('users_2022_2_labeled.csv', index=False)
centers_kmeans6.to_csv('centers_kmeans6.csv', index=False)
#%%
# Generem model amb k=6 i dades sense estandaritzar
kmeans6 = KMeans(n_clusters=6,init='k-means++')
kmeans6.fit(X)
# guardem les etiquetes
X['kmeans6'] = kmeans6.labels_
centers_kmeans6 = pd.DataFrame(kmeans6.cluster_centers_, columns=X.iloc[:,:-1].columns)
sil_k6 = metrics.silhouette_score(X, kmeans6.labels_, metric='euclidean')

#%% guardem dades etiquetades
X.to_csv('users_2022_2_labeled.csv', index=False)
centers_kmeans6.to_csv('centers_kmeans6.csv', index=False)
