# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:00:03 2020

@author: seba3
"""

"""
IMPORTANTE:
    
Prima bisogna eseguire l'intero programma per poter leggere il file con i dati (almeno sul mio pc...)

source: https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/

PROBLEMA:
1 - Penso che sia normale per il simple exponential smoothing avere come predizione una linea retta.
2 - Bisogna aggiungere gli intervalli di confidenza per le previsioni.
3 - Exponential smoothing ha una "discesa anomala". Ho provato a risolvere sommando il risultato del
    simple exponential smoothing ma non è soddisfacente.
4 - Non funzionano le varianti con "multiplicative" invece di "additive" perchè i valori della serie
    devono essere tutti positivi (e lo sono...)
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

if __name__ == "__main__":

    # Costanti per grandezza testo
    
    SMALL_SIZE = 28 
    MEDIUM_SIZE = 30 
    BIGGER_SIZE = 32 
     
    # Inizializzazione caratteristiche base dei PLOT
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes 
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels 
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels 
    plt.rc('legend', fontsize=SMALL_SIZE)    # fontsize of the legend
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # COLORI
    
    TSC = 'black'   # training set
    VSC = 'black'   # validation set
    FC = 'red'      # previsioni
    MRC = 'green'   # model results
    OLC = 'orange'  # other lines
    
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
    
    # usiamo solo la serie maglie. Il procedimento si può ripetere con ciascun capo...
    
    ts = data['MAGLIE'] 
        
    # Se si vuole togliere il 29 febbraio 2016 per avere solo anni di 365 giorni. 
    # Sconsigliato se si considera una stagionalità settimanale in quanto sfalsa di un giorno.
    """
    ts = ts.drop(labels=[pd.Timestamp('2016-02-29')])
    print(ts_totale['2016-02'])
    """
    
    train = ts[pd.date_range(start=ts.index[0], end=ts.index[int(len(ts) * 0.8)], freq='D')]
    valid = ts[pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')]
    
    # Stampa la serie iniziale con suddivisione train e validation (solo per controllo) + rolling mean e std
    # Finestra temporale di un anno per calcolare media e std in movimento
    """
    rolmean = ts.rolling(window=year).mean()
    rolstd = ts.rolling(window=year).std()
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Serie Maglie: training set, validation set, moving average e std')
    plt.ylabel('#Maglie vendute')
    plt.xlabel('Data')
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(rolmean, color=OLC, label='Rolling Mean',  linewidth=3)
    plt.plot(rolstd, color=OLC, label='Rolling Std', linestyle = '--',  linewidth=3)
    plt.legend(loc='best')
    plt.show(block=False)
    plt.plot()
    """
    #%%
    
    #SIMPLE EXPONENTIAL SMOOTHING... risultato non soddisfacente
    
    # create class
    
    modelv1 = SimpleExpSmoothing(train)
    
    # fit model
    
    modelv1_results = modelv1.fit()
    
    # make prediction. Stesso periodo del validation set!
    
    modelv1_predictions = modelv1_results.forecast(steps = int(len(valid)))
    
    #model_predictions = model_results.predict(start="2018-06-11", end="2019-09-29")
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(modelv1_results.fittedvalues, label="Simple Exponential Smoothing", color=MRC)
    plt.plot(modelv1_predictions, label="Forecasts (in sample)", color=FC)
    plt.legend(loc='best')
    plt.plot()
    
    #%%
    
    #EXPONENTIAL SMOOTHING
    
    for i in range(1, len(train)):
        if train[i] < 0:
            train[i] = 0
    
    # Provare a cambiare i parametri per un migliore risultato...
    
    model = ExponentialSmoothing(train, trend="additive", damped = True, seasonal="additive", seasonal_periods=year)
    
    # fit model
    
    model_results = model.fit()
    
    # make prediction. Stesso periodo del validation set!
    
    model_predictions = model_results.forecast(steps = int(len(valid)))
    
    #aggiungo il risultato del simple exponential smoothing (perchè funziona meglio???)
    
    model_predictions = model_predictions + modelv1_predictions
    
    # tolgo i valori negativi
    
    for i in range(1, len(model_predictions)):
        if model_predictions[i] < 0:
            model_predictions[i] = 0
            
    for i in range(1, len(model_results.fittedvalues)):
        if model_results.fittedvalues[i] < 0:
            model_results.fittedvalues[i] = 0
    
    #model_predictions = model_results.predict(start="2018-06-11", end="2019-09-29")
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(model_results.fittedvalues, label="Simple Exponential Smoothing", color=MRC)
    plt.plot(model_predictions, label="Forecasts (in sample)", color=FC)
    plt.legend(loc='best')
    plt.plot()
    
    #%%
    
    errore = model_predictions - valid
    errore.dropna(inplace=True)

    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))


