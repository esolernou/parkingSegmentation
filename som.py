# %% llibreries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('data/users_2022_2_labeled.csv')
users_feb.info()
X = users_feb.iloc[:, 1:-3]

scaler = StandardScaler()
scaler.fit(X)
#X_scaled = pd.DataFrame(scaler.transform(X)).sample(10000).to_numpy()
X_scaled = pd.DataFrame(scaler.transform(X)).to_numpy()

# %% estimació del número de clústers
iterations = 20000
sigma = 1    # neighborhood radius
learning_rate = 0.5
K = range(1, 20)
quantization_errors = []
topographic_errors = []

for k in K:
    som = MiniSom(1, k, 9, sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian', random_seed=10)
    som.train_batch(X_scaled, num_iteration=iterations, verbose=True)
    quantization_errors.append(som.quantization_error(X_scaled))
    topographic_errors.append(som.topographic_error(X_scaled))

# %%
plt.plot(K, quantization_errors, 'gx-')
plt.xlabel('k')
plt.ylabel('quantization error')
plt.title('Elbow Method for SOM Maps (X_scaled)')
plt.show()
# %%
plt.plot(K, topographic_errors, 'bx-')
plt.xlabel('k')
plt.ylabel('topographic errors')
plt.title('Elbow Method for SOM Maps (X_scaled)')
plt.show()

# %%
som_shape = (1, 7)
som = MiniSom(1, k, 9, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function='gaussian', random_seed=10)
# define SOM:
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=9, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(X_scaled, num_iteration=iterations, verbose=True)

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in X_scaled]).T

# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(X_scaled[cluster_index == c, 0],
                X_scaled[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                s=8, linewidths=3, color='k', label='centroid')
plt.legend();
plt.title('SOM Map for k=7')
plt.show()

# %% guardem les etiquetes
users_feb['somK7'] = cluster_index
users_feb.to_csv('data/users_2022_2_labeled.csv', index=False)
pd.DataFrame(scaler.inverse_transform(centroid), columns=users_feb.iloc[:, :-3].columns). \
    to_csv('data/centers_som7.csv', index=False)

#%%
from pylab import plot, axis, show, pcolor, colorbar, bone

bone()
pcolor(som.distance_map().T)  # Distance map as background
colorbar()
show()
