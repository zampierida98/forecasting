# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:32:57 2020

@author: seba3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import mytools as mt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# caricamento insieme dati e verifica tipo delle colonne

data = pd.read_csv('./Dati_Albignasego/Whole period.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)

# L'insieme di dati contiene la data e il numero di capi di abbigliamento venduti
# in quel giorno (per tipo).
# Faccio in modo che la data sia letta per ottenere una serie temporale

dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('./Dati_Albignasego/Whole period.csv', index_col=0, date_parser=dateparse)
#print('\n Date ordinate:')
#print(data.head())
#print(data.index)

ts_maglie = data['MAGLIE']
ts_camicie = data['CAMICIE']
ts_gonne = data['GONNE']
ts_pantaloni = data['PANTALONI']
ts_vestiti = data['VESTITI']
ts_giacche = data['GIACCHE']

ts_totale = ts_maglie + ts_camicie + ts_gonne + ts_pantaloni + ts_vestiti + ts_giacche
print(ts_totale.head())

#print(ts_maglie.head(10),'\n')

#Per ora mi occupo del totale vendite
ts = ts_totale #solo per comodità nella manipolazione dei dati...
plt.figure(figsize=(40, 20), dpi=80)

plt.title(label='Serie temporale iniziale')
plt.ylabel('#Capi venduti')
plt.xlabel('Data')
plt.plot(ts)

#%%
#Test per constatare la stazionarietà di una serie
#Grafici di autocorrelazione e autocorrelazione parziale

mt.test_stationarity(ts, 12, True)
mt.ac_pac_function(ts)

#%%
#Seasonal differencing. E' stata ignorata la presenza dell'anno bisestile...

ts_diff = mt.differencing(ts, 365)
mt.test_stationarity(ts_diff, 365, True)
mt.ac_pac_function(ts_diff)

#%%

ts_diff2 = mt.differencing(ts_diff)
mt.test_stationarity(ts_diff2, 365, True)
mt.ac_pac_function(ts_diff2)

#%% Work in progress
train_set, test_set= np.split(ts, [int(.67 *len(ts))])
p, q = mt.p_q_for_ARIMA(ts)
print(p, q) #1 1
# fit model
model = SARIMAX(train_set, order=(1, 2, 1))
model_fit = model.fit(disp=False)
# make prediction
prediction = model_fit.predict(len(train_set), len(ts), typ='levels')
#%%
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts, color='black')
plt.plot(prediction, color='red')



