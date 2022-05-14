#%% llibreries
import pandas as pd
import numpy as np
import datetime as dt
#%%
dades_feb: pd.DataFrame = pd.read_csv('data/history_2022_02_preprocessed.csv')
dades_feb = dades_feb.drop('Unnamed: 0', axis=1)
# revisió del tipus de dades
dades_feb.info()


#%% Convertim en categòriques les variables següents
categorical = ['BE_N_CAT', 'NB_TOTPAI', 'T_FAMTIT', 'PARKING', 'MONTH', 'WEEKDAY', 'ENTRY_HOUR', 'EXIT_HOUR']
for i, cat in enumerate(categorical):
    dades_feb[cat] = dades_feb[cat].astype('category')

dades_feb.info()
#%% Examinem els casos en que AMOUNT és nul i DISCOUNT NO
aux = dades_feb[dades_feb['AMOUNT'].isna() & ~dades_feb['DISCOUNT'].isna()]

#%%
# registres amb matrícula d'entrada = '???'
dades_feb[dades_feb.ENTRY_PLATE=='2d86c2a659e364e9abba49ea6ffcd53dd5559f05']
# index al dataset
index_entry = dades_feb[dades_feb.ENTRY_PLATE=='2d86c2a659e364e9abba49ea6ffcd53dd5559f05'].ENTRY_PLATE.index
# posem nul el camp matrícula en els índex amb matrícula '???'
for elem in index_entry:
    dades_feb['ENTRY_PLATE'][elem] = ""

# ídem per la matrícula de sortida '???'
# index al dataset
index_exit = dades_feb[dades_feb.EXIT_PLATE=='2d86c2a659e364e9abba49ea6ffcd53dd5559f05'].EXIT_PLATE.index
# posem nul el camp matrícula en els índex amb matrícula '???'
for elem in index_exit:
    dades_feb['EXIT_PLATE'][elem] = ""

#%%

#%%
#dades_feb.to_csv('data\history_2022_02_to_powerbi.csv', index=False, decimal='.')
#dades_feb: pd.DataFrame = pd.read_csv('data\history_2022_02_to_powerbi.csv')
#%%
dades_feb.AMOUNT.describe()
dades_feb.DISCOUNT.describe()
dades_feb.WEEKDAY.describe()

#%%
# NOU CONJUNT DE DADES USERS

# mostra
#aux = dades_feb.sample(200).reset_index()
aux = dades_feb

# simplificació dades_feb
aux = aux.drop(columns=['TIQUET','AMOUNT', 'BE_N_CAT', 'BS_N_CAT', 'EXIT_DT', 'EXIT_HOUR',\
                        'ID_HISTORY', 'MONTH', 'NB_TOTPAI', 'N_CAT'])
#%%
# llistat de matrícules d'entrada úniques
plates_u = np.unique(aux['ENTRY_PLATE'].dropna().to_numpy())

# referència per calcular durada
date_ref = dt.datetime.now()

# discount posem 0 als registres que no han passat pel caixer
aux['DISCOUNT'] = aux['DISCOUNT'].fillna(0)
# %%  PREPARACIÓ VARIABLES CATEGÒRIQUES
aux['CENTRIC_STAY']= ""
aux['MORN_STAY']=""
aux['AFT_STAY'] = ""
aux['NIGHT_STAY'] = ""
aux['LABO'] = ""
aux['SAT'] = ""
aux['SUN'] = ""

for i in range(len(aux)):
    print('\r', i)
    aux['CENTRIC_STAY'][i] = 1 if aux['PARKING'][i] in [1,2,3,4] else 0
    aux['MORN_STAY'][i] = 1 if aux['ENTRY_HOUR'][i] in [6,7,8,9,10,11,12] else 0
    aux['AFT_STAY'][i] = 1 if aux['ENTRY_HOUR'][i] in [13,14,15,16,17,18,19] else 0
    aux['NIGHT_STAY'][i] = 1 if aux['ENTRY_HOUR'][i] in [20,21,22,23,0,1,2,3,4,5] else 0
    aux['LABO'][i] = 1 if aux['WEEKDAY'][i] in ['Dilluns', 'Dimarts', 'Dimecres', 'Dijous', 'Divendres'] else 0
    aux['SAT'][i] = 1 if aux['WEEKDAY'][i] == 'Dissabte' else 0
    aux['SUN'][i] = 1 if aux['WEEKDAY'][i] == 'Diumenge' else 0


#%%
users_feb = aux.groupby(by=['ENTRY_PLATE'], as_index=False).agg(\
    total_stays = ('T_FAMTIT', 'count'),
    centric_stays = ('CENTRIC_STAY', 'sum'),
    duration = ('STAY_HOURS', 'mean'),
    morn_stays = ('MORN_STAY', 'sum'),
    aft_stays = ('AFT_STAY', 'sum'),
    night_stays = ('NIGHT_STAY', 'sum'),
    stays_labo = ('LABO', 'sum'),
    stays_sat = ('SAT', 'sum'),
    stays_sun = ('SUN', 'sum'),
    recency = ('ENTRY_DT', lambda x: (date_ref - pd.to_datetime(max(x))).days),
    discount = ('DISCOUNT', 'mean')
    )

#users_feb.to_csv('users_2022_02.csv', index=False)

#%%
#registres amb matrícula d'entrada nula
aux = dades_feb[dades_feb.ENTRY_PLATE.isna()]
users_feb['total_stays'].sum()+len(aux)    # comprovació del total de registres = len(dades_feb)