# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:00:03 2020

@author: seba3
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import mytools as mt

# Costanti per grandezza testo

SMALL_SIZE = 28 
MEDIUM_SIZE = 30 
BIGGER_SIZE = 32 
 
# Inizializzazione caratteristiche base dei PLOT

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# COLORI

TSC = 'black' # training set
VSC = 'black' # validation set
FC = 'red' # previsioni
MRC = 'green' # model results
OLC = 'orange' # other lines

# STAGIONI

year = 365 # giorni
week = 7

# caricamento insieme dati e verifica tipo delle colonne (solo per controllo)
"""
data = pd.read_csv('./Dati_Albignasego/Whole period.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)
"""
# L'insieme di dati contiene la data e il numero di capi di abbigliamento venduti
# in quel giorno (per tipo).
# Faccio in modo che la data sia letta per ottenere una serie temporale

dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('./Dati_Albignasego/Whole period.csv', index_col=0, date_parser=dateparse)

ts = data['MAGLIE'] # usiamo solo la serie maglie. Il procedimento si può ripetere con ciascun capo...
    
# Se si vuole togliere il 29 febbraio 2016 per avere solo anni di 365 giorni. 
# Sconsigliato se si considera una stagionalità settimanale in quanto sfalsa di un giorno.
"""
ts = ts.drop(labels=[pd.Timestamp('2016-02-29')])
print(ts_totale['2016-02'])
"""

train = ts[pd.date_range(start=ts.index[0], end=ts.index[int(len(ts) * 0.8)], freq='D')]
valid = ts[pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')]

# Stampa la serie iniziale con suddivisione train e validation (solo per controllo)
"""
plt.figure(figsize=(40, 20), dpi=80)
plt.title('Serie Maglie: Training set + Validation set')
plt.ylabel('#Maglie vendute')
plt.xlabel('Data')
plt.plot(train, label="training set", color='black')
plt.plot(valid, label="validation set", color = 'black', linestyle = '--')
plt.legend(loc='best')
plt.plot()
"""


