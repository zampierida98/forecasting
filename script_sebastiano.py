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

plt.figure(num=1, figsize=(40, 40), dpi=80, facecolor='w', edgecolor='k')
plt.plot(ts)
plt.ylabel('#Capi venduti')
plt.xlabel('Data')
plt.show()

#Test per constatare la stazionarietà di una serie

mt.test_stationarity(ts, 12, True, 311)

mt.ac_pac_function(ts, 312, 313)

