# %% llibreries
import pandas as pd
import datetime as dt

# %%  eliminació matrícula
# dades_22_02 = pd.read_csv('data/history_2022_02.csv')
# dades_22_02 = dades_22_02.drop(columns=['BE_N_LICPLA', 'BS_N_LICPLA'])
# dades = dades_22_02.drop(columns=['Unnamed: 0'])
# dades.to_csv('history_origin.csv', index=False)
# %%  LECTURA DE DADES
dades = pd.read_csv('data/history_origin.csv')
# %%  noves variables calculades
dades['STAY_HOURS'] = ""
dades['MONTH'] = ""
dades['WEEKDAY'] = ""
dades['ENTRY_HOUR'] = ""
dades['EXIT_HOUR'] = ""

#%% diccionari weekdays
weekdays = {1: "Dilluns",
            2: "Dimarts",
            3: "Dimecres",
            4: "Dijous",
            5: "Divendres",
            6: "Dissabte",
            7: "Diumenge"}
# %%
# noves variables calculades
dades['EXIT_DT'] = pd.to_datetime(dades['EXIT_DT'])
dades['ENTRY_DT'] = pd.to_datetime(dades['ENTRY_DT'])

volatile_data = {'STAY_HOURS': '', 'MONTH': '', 'WEEKDAY': '', 'ENTRY_HOUR': '', 'EXIT_HOUR': ''}
for i in range(len(dades)):
    print('\r', i)
    current_entry = dades['ENTRY_DT'][i]
    current_exit = dades['EXIT_DT'][i]

    volatile_data['MONTH'] = current_entry.month
    volatile_data['WEEKDAY'] = current_entry.isoweekday()
    volatile_data['ENTRY_HOUR'] = current_entry.hour
    volatile_data['EXIT_HOUR'] = current_exit.hour

    dades['STAY_HOURS'][i] = volatile_data['STAY_HOURS']
    dades['MONTH'][i] = volatile_data['MONTH']
    dades['WEEKDAY'][i] = weekdays[volatile_data['WEEKDAY']]
    dades['ENTRY_HOUR'][i] = volatile_data['ENTRY_HOUR']
    dades['EXIT_HOUR'][i] = volatile_data['EXIT_HOUR']

    if (current_entry != 0) & (current_exit != 0):
        dades['STAY_HOURS'][i] = (current_exit- current_entry).total_seconds() / 3600  # durada en hores quan tenim entrada i sortida

# %%
#dades.to_csv('history_2022_preprocessed.csv')
#dades.to_csv('history_2022_02_preprocessed_.csv')


# %%
dades.info()
