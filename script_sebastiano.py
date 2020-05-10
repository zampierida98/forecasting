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

season = 365 #giorni

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
ts_totale = ts_maglie + ts_camicie + ts_gonne + ts_pantaloni + ts_vestiti + ts_giacche
"""
for i in range(1, len(ts_totale)):
    ts_totale[i] = ts_totale[i]+1
"""    
#ts_totale = ts_totale.drop(labels=[pd.Timestamp('2016-02-29')])
#print(ts_totale['2016-02'])

train = ts_totale[pd.date_range(start=ts_totale.index[0], end=ts_totale.index[int(len(ts_totale) * 0.8)], freq='D')]

#Mi occupo del totale vendite

#Elimino il 29 febbraio 2016 per avere sempre periodi di 365 giorni.
train.drop(labels=[pd.Timestamp('2016-02-29')])
ts = train #solo per comodità nella manipolazione dei dati...

plt.figure(figsize=(40, 20), dpi=80)
plt.title(label='Serie temporale iniziale')
plt.ylabel('#Capi venduti')
plt.xlabel('Data')
plt.plot(ts)

#Test per constatare la stazionarietà di una serie
mt.test_stationarity(ts, season, True)
plt.title(label = "Serie iniziale (totale capi venduti)")

#Grafici di autocorrelazione e autocorrelazione parziale
#mt.ac_pac_function(ts)
plt.title(label = "Serie iniziale (totale capi venduti)")

#ts è la serie con il tale di capi venduti + 1 per giorno da cui è stato tolto il 29 febbraio 2016
ts_log = np.sqrt(ts)
#mt.test_stationarity(ts_log, season, True)

#Differenziazione e prove per controllare il funzionamento di cumulative sums...

ts_log_diff1 = ts_log.diff()
ts_log_diff1.dropna(inplace = True)
mt.test_stationarity(ts_log_diff1, season, True)
mt.kpss_test(ts_log_diff1)
mt.ac_pac_function(ts_log_diff1)

"""
#Controllo funzionamento
print("\nts_log:")
print(ts_log.head())
print("\nts_log_diff:")
print(ts_log_diff1.head())
print("\nts_log_diff2:")
print(ts_log_diff2.head())

#Problema del 29 febbraio...
restored.iloc[season:] = np.nan
for d, val in ts_log_diff.iloc[season:].iteritems():
    restored[d] = restored[d - pd.DateOffset(days=season)] + val

#Risolto problema 29 febbraio!
restored = ts_log.copy()
restored.iloc[season:] = np.nan
counter = 0
for d, val in ts_log_diff.iloc[season:].iteritems():
    restored[d] = restored.iloc[counter-season]+val
    counter+=1

#Test:
restored1 = mt.cumulative_sums(ts_log_diff2, season, ts_log_diff1)
restored2 = mt.cumulative_sums(restored1, season, ts_log)
print("\nRestored1:")
print(restored1.head())
print("\nRestored2:")
print(restored2.head())
"""
#%%
mt.ac_pac_function(ts_log_diff1)
#p, q = mt.p_q_for_ARIMA(ts_log_diff1)
#print(p, q)
p, q = 2, 2
model = ARIMA(ts_log, order=(p, 1, q))
results_ARIMA = model.fit(disp=-1)  
#plt.plot(ts_diff2_log, color='blue', label='2-differenced logged serie')

plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts_log_diff1, label = "Serie trasformata")
plt.plot(results_ARIMA.fittedvalues, color='red', label='ARIMA')
plt.title("Arima (%d, 1, %d)"% (p, q))
plt.legend(loc='best');
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff1)**2))

#%%

predizione_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)

#riporto a originale
predizione_ARIMA_diff_cumsum = predizione_ARIMA_diff.cumsum()
#print(predizione_ARIMA_diff_cumsum.head())
#predizione_ARIMA_log = pd.Series(ts_log, index = ts_log.index)
predizione_ARIMA_log = pd.Series(predizione_ARIMA_diff_cumsum)
#predizione_ARIMA_log = predizione_ARIMA_log.add(predizione_ARIMA_diff_cumsum, fill_value=0)

#%%
predizione_ARIMA = np.square(predizione_ARIMA_log)
"""
for i in range(1, len(predizione_ARIMA_log)):
    predizione_ARIMA_log[i] = predizione_ARIMA_log[i]-1
"""
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts, color = 'blue', label = 'serie iniziale')
plt.plot(predizione_ARIMA, label ="arima trasformato all' indietro", color = 'red')
plt.title('RMSE: %.4f'% np.sqrt(sum((predizione_ARIMA-ts)**2)/len(ts)))
plt.legend(loc='best');

#%%

#divido in 'train' e 'validation' sets i miei dati
train = ts[pd.date_range(start = ts.index[0], end = ts.index[int(0.8*(len(ts)))])] #80% per training
valid = ts[pd.date_range(start = ts.index[int(0.8*(len(ts)))], end = ts.index[int(len(ts))-1])] #20% per validation

#plotting the data
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(train)
plt.plot(valid)

model = ARIMA(train, order =(2, 1, 2))
model = model.fit()
print(model.summary())

forecast = model.predict(int(len(train)), int(len(train)))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

forecast.dropna(inplace=True)
plt.figure()
plt.plot(forecast)
print(forecast.head())
#%%
#plot the predictions for validation set
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(train, label='Train', color = 'blue')
plt.plot(valid, label='Valid', color = 'yellow')
plt.plot(forecast, label='Prediction', color = 'green')
