# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:32:57 2020

@author: seba3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# caricamento insieme dati e verifica tipo delle colonne

data = pd.read_csv('./Dati_Albignasego/wholeperiod.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)

# L'insieme di dati contiene la data e il numero di capi di abbigliamento venduti
# in quel giorno (per tipo).
# Faccio in modo che la data sia letta per ottenere una serie temporale

dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('./Dati_Albignasego/wholeperiod.csv', parse_dates=['DATA'], index_col=['DATA'],date_parser=dateparse)
print('\n Date ordinate:')
print(data.head())
print(data.index)

ts_maglie = data['MAGLIE']
ts_camicie = data['CAMICIE']
ts_gonne = data['GONNE']
ts_pantaloni = data['PANTALONI']
ts_vestiti = data['VESTITI']
ts_giacche = data['GIACCHE']
#print(ts_maglie.head(10),'\n')

#Per ora mi occupo solo di MAGLIE
ts = ts_maglie #solo per comodità nella manipolazione dei dati...

plt.plot(ts)
plt.ylabel('#Maglie')
plt.xlabel('Data')
plt.show()

#%%

#Test per constatare la stazionarietà di una serie

mt.test_stationarity(ts, 12, True)

#%%

#Rendo la serie stazionaria (2 metodi differenti)

ts_stazionaria_stagionale = mt.make_seasonal_stationary(ts, 12, False)
print('Dopo trasformazione Moving Average:\n')
mt.test_stationarity(ts_stazionaria_stagionale, 12, True)

ts_stazionaria_esponenziale = mt.make_exponential_stationary(ts, 12, False)
print('Dopo trasformazione Exponentially Weighted Moving Average:\n')
mt.test_stationarity(ts_stazionaria_esponenziale, 12, True)

ts_decomposta = mt.decompose(ts)
print('Dopo decomposizione (residui):\n')
mt.test_stationarity(ts_decomposta, 12, True)

#%%

mt.ac_pac_function(ts_stazionaria_esponenziale)