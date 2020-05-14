# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:32:57 2020

@author: seba3
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import mytools as mt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

SMALL_SIZE = 28 
MEDIUM_SIZE = 30 
BIGGER_SIZE = 32 
 
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

TRAINING_SET_C = 'cornflowerblue'
VALIDATION_SET_C = 'gold'
FORECASTS_C = 'black'
MODEL_RESULTS_C = 'red'

season = 182 #giorni

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

for i in range(1, len(ts_totale)):
    ts_totale[i] = ts_totale[i]+1
    
#ts_totale = ts_totale.drop(labels=[pd.Timestamp('2016-02-29')])
#print(ts_totale['2016-02'])

train = ts_totale[pd.date_range(start=ts_totale.index[0], end=ts_totale.index[int(len(ts_totale) * 0.8)], freq='D')]
valid = ts_totale[pd.date_range(start=ts_totale.index[int(len(ts_totale)*0.8)+1], end = ts_totale.index[int(len(ts_totale))-1], freq='D')]
#Mi occupo del totale vendite

#Elimino il 29 febbraio 2016 per avere sempre periodi di 365 giorni.
train = train.drop(labels=[pd.Timestamp('2016-02-29')])
ts = train #solo per comodità nella manipolazione dei dati...

plt.figure(figsize=(40, 20), dpi=80)
plt.title('Training set + Validation set')
plt.ylabel('#Capi venduti')
plt.xlabel('Data')
plt.plot(ts, label="training set")
plt.plot(valid, label="validation set")
plt.legend(loc='best')
plt.plot()

#Test per constatare la stazionarietà di una serie
mt.test_stationarity(ts, season, True)
plt.title(label = "Serie iniziale (totale capi venduti)")
#Grafici di autocorrelazione e autocorrelazione parziale
#mt.ac_pac_function(ts)
plt.title(label = "Serie iniziale (totale capi venduti)")

#ts è la serie con il tale di capi venduti + 1 per giorno da cui è stato tolto il 29 febbraio 2016
#ts_log = np.log(ts)
ts_log = ts
#mt.test_stationarity(ts_log, season, True)
#Differenziazione e prove per controllare il funzionamento di cumulative sums...
ts_log_diff1 = ts_log.diff(periods=365)
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
p, q = mt.p_q_for_ARIMA(ts_log_diff1)
print(p, q)
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

predizione_ARIMA_diff = pd.Series(data = results_ARIMA.fittedvalues, copy = True)

#riporto a originale
predizione_ARIMA_diff_cumsum = predizione_ARIMA_diff.cumsum()
predizione_ARIMA_log = pd.Series(ts_log.iloc[0], index = ts_log.index)
predizione_ARIMA_log = predizione_ARIMA_log.add(predizione_ARIMA_diff_cumsum, fill_value = 0)

#%%
#predizione_ARIMA = np.exp(predizione_ARIMA_log)
"""
for i in range(1, len(predizione_ARIMA_log)):
    predizione_ARIMA_log[i] = predizione_ARIMA_log[i]-1
"""
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts, label = 'serie iniziale')
plt.plot(predizione_ARIMA_log, label ="arima trasformato all' indietro", color = 'red')
plt.title('RMSE: %.4f'% np.sqrt(sum((predizione_ARIMA_log-ts)**2)/len(ts)))
plt.legend(loc='best');


#%%
p, q = 4, 3

my_ts = ts_log.diff(periods=365)
#my_ts = ts_log.diff()
my_ts.dropna(inplace=True)
my_stagionalita = ts_log.rolling(window=15).mean()

model = ARIMA(my_ts, order=(p, 0, q))
results_ARIMA = model.fit(disp=0)  
#plt.plot(ts_diff2_log, color='blue', label='2-differenced logged serie')

my_arima = pd.Series(results_ARIMA.fittedvalues, copy=True)
original_scale = my_arima + my_stagionalita
for i in range(1, len(original_scale)):
    if original_scale[i] < 0:
        original_scale[i] = 0
        
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts_log, label = "Training set")
plt.plot(original_scale, color='orange', label='ARIMA')
plt.title("Arima (%d, 1, %d)"% (p, q))
plt.legend(loc='best');

#%%

predictions, _, interval = results_ARIMA.forecast(steps = int(len(valid)))
predictions = pd.Series(predictions, index=pd.date_range(start=ts_totale.index[int(len(ts_totale)*0.8)+1], periods=int(len(valid)), freq='D'))
predictions.dropna(inplace=True)
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(train, label='training set')
plt.plot(valid, color='c', label='validation set')
plt.plot(original_scale, color = 'orange', label = 'risultati di ARIMA')
forecast = predictions+ts_totale.rolling(window=15).mean()
forecast.dropna(inplace = True)
plt.plot(forecast, color="purple", label='previsione con ARIMA')
plt.title('Previsioni con ARIMA(4,1,3)')
plt.xlabel('Data')
plt.ylabel('#Capi venduti')
plt.legend(loc='best')
print(predictions.head())