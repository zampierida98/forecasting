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

ts_totale = ts_maglie

train = ts_totale[pd.date_range(start=ts_totale.index[0], end=ts_totale.index[int(len(ts_totale) * 0.8)], freq='D')]
valid = ts_totale[pd.date_range(start=ts_totale.index[int(len(ts_totale)*0.8)+1], end = ts_totale.index[int(len(ts_totale))-1], freq='D')]

ts = train #solo per comodità nella manipolazione dei dati...

plt.figure(figsize=(40, 20), dpi=80)
plt.title('Serie Maglie: Training set + Validation set')
plt.ylabel('#Maglie vendute')
plt.xlabel('Data')
plt.plot(ts, label="training set", color='black')
plt.plot(valid, label="validation set", color = 'black', linestyle = '--')
plt.legend(loc='best')
plt.show()

#Test per constatare la stazionarietà di una serie
mt.test_stationarity(ts, season, True)
print("Calcolo in corso...")
#%%

sarima_model = SARIMAX(train, order=(3,1,2), seasonal_order=(1,1,2,7), enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit(disp = -1, maxiter = 200)

sarima_pred = sarima_fit.predict(start="2018-06-11", end="2019-09-29")

for i in range(1, len(sarima_pred)):
    if sarima_pred[i] < 0:
        sarima_pred[i] = 0
        
for i in range(1, len(sarima_fit.fittedvalues)):
    if sarima_fit.fittedvalues[i] < 0:
        sarima_fit.fittedvalues[i] = 0

predint_xminus = pd.Series.copy(valid)
predint_xplus  = pd.Series.copy(valid)

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
plt.show()