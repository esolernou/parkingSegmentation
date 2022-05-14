#%% llibreries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm
import paramnormal
import numpy as np

#%%
users_feb: pd.DataFrame = pd.read_csv('data/users_2022_02.csv')
#users_feb = users_feb.drop('Unnamed: 0', axis=1)
users_feb['ENTRY_PLATE'] = users_feb['ENTRY_PLATE'].astype('category')
#%%
users_describe = users_feb.describe()
users_feb.info()

# eliminem usuari amb ENTRY_PLATE = nan
# users_feb = users_feb[~users_feb['ENTRY_PLATE'].isna()]
# users_feb = users_feb[~users_feb['duration'].isna()]
# users_describe = users_feb.describe()
users_feb.info()


#%% Eliminem vehicles d'empresa
# manteniment = ['3b375a8f4b3028ec71a03f966931298fa820289a', '582666fcfd3786723b8f86ddbe0d3069f31c54e7',\
#           'dee8b858e1c4c177f72b4672d6b877d32a8ee58b','cad638be92d2270c58dec78233cfaf2a42108faf',\
#                '3467ed62b2c29cb4da14cd19fc469385014cb178','797ba3d1906f73f0041474c618a6c3559cde22b8',\
#                '649f4bf5e62af706780622f95618152347a187b8']
#
#
# users_feb = users_feb[~users_feb['ENTRY_PLATE'].isin(manteniment)]
#
# # actualitzem l'arxiu
# users_feb.to_csv('data/users_2022_02.csv', index=False)
#%%

# GRÀFIQUES AL NOTEBOOK

# registres amb duration nul·la
aux = users_feb[users_feb['duration'].isna()]

#%%
users_feb.hist(bins = 50, figsize=(30,20))
plt.show()

users_feb.describe().T
#%% funció de probabilitat
sns.displot(users_feb['total_stays'], kde=True)
plt.show()

#%%   --------- DURATION -----------------
# lognormal fit
x = users_feb[users_feb['duration']<25]['duration']
fitting_params = paramnormal.lognormal.fit(x)
print(fitting_params)
lognorm_dist = paramnormal.lognormal.from_params(fitting_params)
#%%
sns.lineplot(x=t,y=lognorm_dist.pdf(t), lw=2, color='r',\
            label='Fitted Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(fitting_params[0], fitting_params[1]))
plt.show()

#%%
# generem eix x
t = np.linspace(np.min(x), np.max(x), 100)
max_dur = max(users_feb[users_feb['duration']<25]['duration'])
f, ax = plt.subplots(1, figsize=(10, 5))
sns.displot(users_feb[users_feb['duration']<25]['duration'], kind='kde', label='Kernel density estimate of the data ')
sns.lineplot(x=t,y=lognorm_dist.pdf(t), lw=2, color='r',\
            label='Fitted Model X~LogNorm(mu={0:.2}, sigma={1:.2f})'.format(fitting_params[0], fitting_params[1]))
plt.legend(loc='upper right', fontsize=8)
plt.title('DURATION', fontsize=10)
plt.show()

#%%  -------------- TOTAL STAYS ---------------
# lognormal fit
x = users_feb[users_feb['total_stays']<70]['total_stays']
fitting_params = paramnormal.lognormal.fit(x)
print(fitting_params)
lognorm_dist = paramnormal.lognormal.from_params(fitting_params)

#%%
# generem eix x
t = np.linspace(np.min(x), np.max(x), 100)
f, ax = plt.subplots(1, figsize=(10, 5))
sns.displot(x, kind='kde', label='Kernel density estimate of the data ')
sns.lineplot(x=t,y=lognorm_dist.pdf(t), lw=2, color='r',\
            label='Fitted Model X~LogNorm(mu={0:.2}, sigma={1:.2f})'.format(fitting_params[0], fitting_params[1]))
plt.legend(loc='upper right', fontsize=8)
plt.title('TOTAL STAYS', fontsize=10)
plt.show()

#%%