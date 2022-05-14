# %% llibreries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('data/users_2022_2_labeled.csv')
users_feb.info()
X = users_feb.iloc[:, :-1]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X)).sample(10000)
X_scaled.info()
# %%
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# %%
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
model = model.fit(X_scaled)
plt.title("Hierarchical Clustering Dendrogram - ward linkage")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=8)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# %%
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete')
model = model.fit(X_scaled)
plt.title("Hierarchical Clustering Dendrogram - complete linkage")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=8)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
# %%
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average')
model = model.fit(X_scaled)
plt.title("Hierarchical Clustering Dendrogram - average linkage")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=8)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# %%
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single')
model = model.fit(X_scaled)
plt.title("Hierarchical Clustering Dendrogram - single linkage")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=10)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#%%   Model amb ward, k=6 i etiquetat de les dades
X_scaled = pd.DataFrame(scaler.transform(X))
model = AgglomerativeClustering(distance_threshold=None, n_clusters=6, linkage='ward')
model = model.fit(X_scaled)
labels = model.labels_

users_feb['hierarchical']=labels
#%% guardem dades etiquetades
users_feb.to_csv('data/users_2022_2_labeled.csv', index=False)
