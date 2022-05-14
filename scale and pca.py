# %% llibreries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# %% càrrega de dades
users_feb: pd.DataFrame = pd.read_csv('data/users_2022_02.csv')
# eliminem la primera columna (Entry_plate)
X = users_feb.iloc[:, 2:]
X.info()
# %%  normalització dades
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% PCA
pca = PCA()
pca.fit(X_scaled)
principalComponents = pca.transform(X_scaled)

#%%
explained_var = pd.DataFrame({'column':X.columns, 'explained_var':pca.explained_variance_ratio_*100})

#%%
explained_var.to_csv('explained_var.csv')
X.iloc[:,:-1].to_csv('users_feb_pca.csv')