# %% llibreries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns

# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('data/users_2022_2_labeled.csv')
users_feb.info()
X = users_feb.iloc[:, :-1]
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
#%% dbscan
model = DBSCAN(eps=2, min_samples=50, n_jobs=-1)
clusters = model.fit(X_scaled)

labels = clusters.labels_
sns.displot(labels)
plt.show()

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_scaled, labels))

#%% 3D plot of a sample
sample = X[(X['duration']<12) & (X['centric_stays']<7)].sample(2000)
sample_scaled = scaler.transform(sample)

# get cluster of each point
dbscan_result = clusters.fit_predict(sample_scaled)

# clusters
dbscan_clusters = np.unique(dbscan_result)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plt
for dbscan_cluster in dbscan_clusters:
    # get index points
    index = np.where(dbscan_result == dbscan_cluster)
    # plot 3d
    ax.scatter(sample_scaled[index, 0], sample_scaled[index, 1], sample_scaled[index, 2],)

ax.set_xlabel('centric_stays')
ax.set_ylabel('duration')
ax.set_zlabel('morn_stays')

plt.show()

#%% dbscan
model2 = DBSCAN(eps=1, min_samples=50, n_jobs=-1)
clusters = model2.fit(X_scaled)

labels = clusters.labels_
sns.displot(labels)
plt.show()

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_scaled, labels))

#%% 3D plot of a sample
sample = X[(X['duration']<12) & (X['centric_stays']<7)].sample(2000)
sample_scaled = scaler.transform(sample)

# get cluster of each point
dbscan_result = clusters.fit_predict(sample_scaled)

# clusters
dbscan_clusters = np.unique(dbscan_result)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plt
for dbscan_cluster in dbscan_clusters:
    # get index points
    index = np.where(dbscan_result == dbscan_cluster)
    # plot 3d
    ax.scatter(sample_scaled[index, 0], sample_scaled[index, 1], sample_scaled[index, 2])

ax.set_xlabel('centric_stays')
ax.set_ylabel('duration')
ax.set_zlabel('morn_stays')

plt.show()

#%% dbscan
model2 = DBSCAN(eps=0.7, min_samples=30, n_jobs=-1)
clusters = model2.fit(X_scaled)

labels = clusters.labels_
sns.displot(labels)
plt.show()

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X_scaled, labels))

#%% 3D plot of a sample
sample = X[(X['duration']<12) & (X['centric_stays']<7)].sample(2000)
sample_scaled = scaler.transform(sample)

# get cluster of each point
dbscan_result = clusters.fit_predict(sample_scaled)

# clusters
dbscan_clusters = np.unique(dbscan_result)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# plt
for dbscan_cluster in dbscan_clusters:
    # get index points
    index = np.where(dbscan_result == dbscan_cluster)
    # plot 3d
    ax.scatter(sample_scaled[index, 0], sample_scaled[index, 1], sample_scaled[index, 2])

ax.set_xlabel('centric_stays')
ax.set_ylabel('duration')
ax.set_zlabel('morn_stays')

plt.show()