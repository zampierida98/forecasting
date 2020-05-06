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
from statsmodels.tsa.arima_model import ARIMA

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

ts_maglie = data['MAGLIE']
ts_camicie = data['CAMICIE']
ts_gonne = data['GONNE']
ts_pantaloni = data['PANTALONI']
ts_vestiti = data['VESTITI']
ts_giacche = data['GIACCHE']

#togliere i valori nulli per applicare log!
#ts_totale = ts_maglie + ts_camicie + ts_gonne + ts_pantaloni + ts_vestiti + ts_giacche
ts_totale = ts_maglie
for i in range(1, len(ts_totale)):
    ts_totale[i] = ts_totale[i]+1
    
#Elimino il 29 febbraio 2016 per avere sempre periodi di 365 giorni
ts_totale = ts_totale.drop(labels=[pd.Timestamp('2016-02-29')])
#print(ts_totale['2016-02'])

#Per ora mi occupo del totale vendite
ts = ts_totale #solo per comodità nella manipolazione dei dati...

#plt.figure(figsize=(40, 20), dpi=80)
#plt.title(label='Serie temporale iniziale')
#plt.ylabel('#Capi venduti')
#plt.xlabel('Data')
#plt.plot(ts)

#Test per constatare la stazionarietà di una serie
#mt.test_stationarity(ts, 365, True)

#Grafici di autocorrelazione e autocorrelazione parziale
#mt.ac_pac_function(ts)

#ts è la serie con il tale di capi venduti + 1 per giorno da cui è stato tolto il 29 febbraio 2016
ts_log = np.log(ts)
#mt.test_stationarity(ts_log, 365, True)

#Differenziazione

ts_diff2_log = mt.differencing(ts_log, 365)
mt.test_stationarity(ts_diff2_log, 365, True)
mt.ac_pac_function(ts_diff2_log)

#%% Work in progress

#p, q = mt.p_q_for_ARIMA(ts_diff2_log)
p=2
q=8
print(p, q)
model = ARIMA(ts_diff2_log, order=(p, 0, q))
results_ARIMA = model.fit(disp=-1)  
#plt.plot(ts_diff2_log, color='blue', label='2-differenced logged serie')
pd.DataFrame(ts_diff2_log).plot()
plt.plot(results_ARIMA.fittedvalues, color='red', label='ARIMA')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_diff2_log)**2))

predizione_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy = True)

#%%
#riporto a originale

#result = results_ARIMA.fittedvalues + mt.differencing(ts_log, 365).shift(365) + ts_log.shift(365)
result = mt.cumulative_sums(results_ARIMA.fittedvalues, 365)
result.dropna(inplace = True)
result = np.exp(result)
print(result.head())

plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts, color = 'blue', label = 'serie iniziale')
plt.plot(result, color = 'red', label ='arima trasformato all\'indietro')
