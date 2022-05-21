# %% llibreries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('data/users_labeled.csv')
users_feb.info()
X = users_feb.iloc[:, 3:-6]

scaler = StandardScaler()
scaler.fit(X)

X_scaled = pd.DataFrame(scaler.transform(X)).to_numpy()

# %% fraud detection
som_shape = (50, 50)
sigma = 1
learning_rate = 0.8
iterations = 20000

# define SOM:
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=9, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(X_scaled, num_iteration=iterations, verbose=True)

#%%
weights = som.get_weights()
dist_map: pd.DataFrame = som.distance_map()   # higher distance -> BMU far from the clusters

# %%
from pylab import plot, axis, show, pcolor, colorbar, bone

bone()
pcolor(som.distance_map().T)  # Distance map as background
colorbar()
show()

#%% get neuron with max distance
out_position = np.where(som.distance_map()> 0.9)
print(out_position[0],',', out_position[1])
dist_map[30,17]

#%% mapping
mappings = som.win_map(X_scaled)   # winner neuron for each element
som.winner(X_scaled[0])    # winner neuron for an element

mappings.keys()   # coordenades del mapa
len(mappings.keys())

#%%
frauds = mappings[(30,17)]
frauds

# the list of customers who are frauds:
users_fraud= pd.DataFrame(scaler.inverse_transform(frauds), columns = X.columns)

users_fraud.to_csv('users_fraud_50.csv', index=False)

# %% fraud detection
som_shape = (20, 20)
sigma = 1
learning_rate = 0.8
iterations = 20000

# define SOM:
som = MiniSom(x=som_shape[0], y=som_shape[1], input_len=9, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(X_scaled, num_iteration=iterations, verbose=True)

#%%
weights = som.get_weights()
dist_map: pd.DataFrame = som.distance_map()   # higher distance -> BMU far from the clusters

# %%
from pylab import plot, axis, show, pcolor, colorbar, bone

bone()
pcolor(som.distance_map().T)  # Distance map as background
colorbar()
show()

#%% get neuron with max distance
out_position = np.where(som.distance_map()> 0.9)
print(out_position[0],',', out_position[1])

dist_map[3,12]

#%% mapping
mappings = som.win_map(X_scaled)   # winner neuron for each element
som.winner(X_scaled[0])    # winner neuron for an element

mappings.keys()   # coordenades del mapa
len(mappings.keys())

#%%
frauds = mappings[(3,12)]
frauds

# the list of customers who are frauds:
users_fraud= pd.DataFrame(scaler.inverse_transform(frauds), columns = X.columns)

users_fraud.to_csv('users_fraud_20.csv', index=False)