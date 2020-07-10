# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:32:57 2020

@author: seba3
"""

#SARIMA (3,0,2)x(1,1,2,7)

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import mytools as mt
import pmdarima as pm
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

TRAINING_SET_C = 'black'
VALIDATION_SET_C = 'black'
FORECASTS_C = 'red'
MODEL_RESULTS_C = 'green'
OTHER_LINES_C = 'orange'

season = 365 #giorni
season_2 = 7

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

#for i in range(1, len(ts_totale)):
#    ts_totale[i] = ts_totale[i]+1
    
#ts_totale = ts_totale.drop(labels=[pd.Timestamp('2016-02-29')])
#print(ts_totale['2016-02'])

train = ts_totale[pd.date_range(start=ts_totale.index[0], end=ts_totale.index[int(len(ts_totale) * 0.8)], freq='D')]
valid = ts_totale[pd.date_range(start=ts_totale.index[int(len(ts_totale)*0.8)+1], end = ts_totale.index[int(len(ts_totale))-1], freq='D')]
#Mi occupo del totale vendite

#Elimino il 29 febbraio 2016 per avere sempre periodi di 365 giorni.
#train = train.drop(labels=[pd.Timestamp('2016-02-29')])
ts = train #solo per comodità nella manipolazione dei dati...

plt.figure(figsize=(40, 20), dpi=80)
plt.title('Serie Maglie: Training set + Validation set')
plt.ylabel('#Maglie vendute')
plt.xlabel('Data')
plt.plot(ts, label="training set", color='black')
plt.plot(valid, label="validation set", color = 'black', linestyle = '--')
plt.legend(loc='best')
plt.plot()

#Test per constatare la stazionarietà di una serie
mt.test_stationarity(ts, season, True)
plt.title(label = "Serie iniziale (maglie vendute)")
#Grafici di autocorrelazione e autocorrelazione parziale
#mt.ac_pac_function(ts)
plt.title(label = "Serie iniziale (maglie vendute)")
#%%
"""
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA

smodel = pm.auto_arima(train, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=7,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
smodel.summary()
"""

#%%

sarima_model = SARIMAX(train, order=(3,1,2), seasonal_order=(1,1,2,7), enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit(disp = -1, maxiter = 200)

sarima_pred = sarima_fit.predict(start="2018-06-11", end="2019-09-29")
#get_prediction("2018-06-11", "2018-12-31")

for i in range(1, len(sarima_pred)):
    if sarima_pred[i] < 0:
        sarima_pred[i] = 0
        
for i in range(1, len(sarima_fit.fittedvalues)):
    if sarima_fit.fittedvalues[i] < 0:
        sarima_fit.fittedvalues[i] = 0

predint_xminus = ts[pd.date_range(start="2018-06-11", end ="2019-09-29", freq='D')]
predint_xplus  = ts[pd.date_range(start="2018-06-11", end ="2019-09-29", freq='D')]

z = 1.96
sse = sarima_fit.sse
for i in range(1, len(sarima_pred)):
    predint_xminus[i] = sarima_pred[i] - z * np.sqrt(sse/len(predint_xminus)+i)
    predint_xplus[i]  = sarima_pred[i] + z * np.sqrt(sse/len(predint_xplus)+i)

plt.figure(figsize=(40, 20), dpi=80)
plt.plot(train, label = "Training set", color = 'black')
plt.plot(valid, label = "Validation set", color = "black", linestyle = "--")
plt.plot(sarima_fit.fittedvalues, color='green', label='SARIMAX model')
plt.plot(sarima_pred, color="red", label='SARIMAX predictions')
plt.title("Sarimax (3,0,2)x(1,1,2,7)")
plt.fill_between(pd.date_range(start="2018-06-11", periods=len(valid) , freq='D'), 
                 predint_xplus, 
                 predint_xminus, 
                 color='grey', alpha=.25)
plt.legend(loc='best')

# Forecast
"""
n_periods = 7
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(train.index[-1], periods = n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
predint_xminus = pd.Series(confint[:, 0], index=index_of_fc)
predint_xplus = pd.Series(confint[:, 1], index=index_of_fc)

z = 1.96
sse = fitted.sse
for i in range(1, len(valid)):
    predint_xminus[i] = fitted_series[i] - z * np.sqrt(sse/len(valid)+i)
    predint_xplus[i]  = fitted_series[i] + z * np.sqrt(sse/len(valid)+i)

# Plot
plt.plot(train)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(predint_xminus.index, 
                 predint_xminus, 
                 predint_xplus, 
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast")
plt.show()
"""
"""
#%%
#ts è la serie con il tale di capi venduti + 1 per giorno da cui è stato tolto il 29 febbraio 2016
#ts_log = np.log(ts)
ts_log = ts
#mt.test_stationarity(ts_log, season, True)
#Differenziazione e prove per controllare il funzionamento di cumulative sums...
ts_log_diff1 = ts_log.diff(periods=season)
ts_log_diff1.dropna(inplace = True)
mt.test_stationarity(ts_log_diff1, season, True)
mt.kpss_test(ts_log_diff1)
mt.ac_pac_function(ts_log_diff1)

#%%

ts_log_diff2 = ts_log_diff1.diff(periods=season_2)
ts_log_diff2.dropna(inplace = True)
mt.test_stationarity(ts_log_diff2, season_2, True)
mt.kpss_test(ts_log_diff2)
mt.ac_pac_function(ts_log_diff2)

#%%

#weekly_cumulative_sum = ts_log_diff2.rolling(window=7).sum()
#weekly_cumulative_sum.dropna(inplace = True)


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
#NUOVO! provo con sarimax per poter applicare una differenziazione "normale" e una con stagionalità
"""
sarima_model = SARIMAX(ts_log, order=(4, 1, 3), seasonal_order=(2, 0, 2, 7))#, enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit(disp = -1)

sarima_pred = sarima_fit.forecast(steps = int(len(valid)))
#get_prediction("2018-06-11", "2018-12-31")

plt.figure(figsize=(40, 20), dpi=80)
plt.plot(train, label = "Training set", color = 'black')
plt.plot(valid, label = "Validation set", color = "black", linestyle = "--")
plt.plot(sarima_fit.fittedvalues, color='green', label='SARIMAX model')
plt.plot(sarima_pred, color="red", label='SARIMAX predictions')
plt.title("Sarima (%d, 1, %d) x (%d, 1, %d, %d)" %(0, 2, 0, 2, season_2))
plt.legend(loc='best')
"""
#%%
"""
mt.ac_pac_function(ts_log_diff2)
p, q = mt.p_q_for_ARIMA(ts_log_diff2)
print(p, q)
model = ARIMA(ts_log, order=(p, 1, q))
results_ARIMA = model.fit(disp=-1)  
#plt.plot(ts_diff2_log, color='blue', label='2-differenced logged serie')

plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts_log_diff1, label = "Serie trasformata", color = 'black')
plt.plot(results_ARIMA.fittedvalues, color='green', label='ARIMA')
plt.title("Arima (%d, 1, %d)"% (p, q))
plt.legend(loc='best');
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff1)**2))

predizione_ARIMA_diff = pd.Series(data = results_ARIMA.fittedvalues, copy = True)

#riporto a originale
predizione_ARIMA_diff_cumsum = predizione_ARIMA_diff.cumsum()
predizione_ARIMA_log = pd.Series(ts_log.iloc[0], index = ts_log.index)
predizione_ARIMA_log = predizione_ARIMA_log.add(predizione_ARIMA_diff_cumsum, fill_value = 0)
"""
#%%
#predizione_ARIMA = np.exp(predizione_ARIMA_log)
"""
for i in range(1, len(predizione_ARIMA_log)):
    predizione_ARIMA_log[i] = predizione_ARIMA_log[i]-1

plt.figure(figsize=(40, 20), dpi=80)
plt.plot(ts, label = 'serie iniziale', color = 'black')
plt.plot(predizione_ARIMA_log, label ="arima trasformato all' indietro", color = 'green')
plt.title('RMSE: %.4f'% np.sqrt(sum((predizione_ARIMA_log-ts)**2)/len(ts)))
plt.legend(loc='best');
"""

#%%
"""
p, q = 2, 2

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
plt.plot(ts_log, label = "Training set", color = 'black')
plt.plot(original_scale, color='green', label='ARIMA')
plt.title("Arima (%d, 1, %d)"% (p, q))
plt.legend(loc='best');

#%%

predictions, _, interval = results_ARIMA.forecast(steps = int(len(valid)))
predictions = pd.Series(predictions, index=pd.date_range(start=ts_totale.index[int(len(ts_totale)*0.8)+1], periods=int(len(valid)), freq='D'))
predictions.dropna(inplace=True)
plt.figure(figsize=(40, 20), dpi=80)
plt.plot(train, label='training set', color = 'black')
plt.plot(valid, color='black', label='validation set', linestyle = '--')
plt.plot(original_scale, color = 'green', label = 'risultati di ARIMA')
forecast = predictions+ts_totale.rolling(window=15).mean()
forecast.dropna(inplace = True)

for i in range(1, len(forecast)):
    if forecast[i] < 0:
        forecast[i] = 0
 
ci = 1.96 * np.std(forecast)/np.mean(forecast)
plt.plot(forecast, color="red", label='previsione con ARIMA')
plt.title('Previsioni con ARIMA(4,1,3)')
plt.xlabel('Data')
plt.ylabel('#Maglie vendute')
plt.legend(loc='best')
print(predictions.head())

#%%

errore = forecast - valid
errore.dropna(inplace=True)

print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))

"""