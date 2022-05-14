# %% llibreries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import seaborn as sns

# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('../data/users_2022_02.csv')
users_feb.info()
X = users_feb.iloc[:, 1:]
X_scaled = StandardScaler().fit_transform(X)

#%%
gm = GaussianMixture(n_components=11, random_state=0,
    covariance_type='full').fit(X)

print('aic: ', gm.aic(X))
print('score: ', gm.score(X))

#%%
gm = GaussianMixture(n_components=11, random_state=0,
    covariance_type='spherical').fit(X)

print('aic: ', gm.aic(X))
print('score: ', gm.score(X))
#%%
bm = BayesianGaussianMixture(n_components=11,
                             covariance_type='full').fit(X)
bm.score(X)
#%%
bm = BayesianGaussianMixture(n_components=11,
                             covariance_type='full').fit(X_scaled)
bm.score(X_scaled)