# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:00:03 2020

@author: seba3
"""

"""
IMPORTANTE:
    
Prima bisogna eseguire l'intero programma per poter leggere il file con i dati (almeno sul mio pc...)

source: https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/

Exponential smoothing ha una "discesa anomala". Ho provato a risolvere sommando il risultato del 
simple exponential smoothing ma non è soddisfacente.
Il simple exponential smoothing fa un forecast piatto ovvero Y_{t+h} = Y_{t+1} = l_t
dove l_t è l'equazione che descrive il livello della serie. Quindi è normale
che le previsioni siano una linea retta

"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

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
    
    fitted = modelv1.fit()
    
    # make prediction. Stesso periodo del validation set!
    
    #forecasted = fitted.forecast(steps = int(len(valid)))
    
    forecasted = fitted.predict(start="2018-06-11", end="2019-09-29")
    
    # Calcolo gli intervalli di predizione
    predint_xminus = ts[pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')]
    predint_xplus  = ts[pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')]
    
    z = 1.96
    sse = fitted.sse
    for i in range(1, len(valid)):
        predint_xminus[i] = forecasted[i] - z * np.sqrt(sse/len(valid)+i)
        predint_xplus[i]  = forecasted[i] + z * np.sqrt(sse/len(valid)+i)
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(fitted.fittedvalues, label="Simple Exponential Smoothing", color=MRC)
    plt.plot(forecasted, label="Forecasts (in sample)", color=FC)
    plt.plot(predint_xminus, color="grey", alpha = .5)
    plt.plot(predint_xplus, color="grey", alpha = .5)
    plt.fill_between(pd.date_range(start="2018-06-11", periods=len(valid) , freq='D'), 
                 predint_xplus, 
                 predint_xminus, 
                 color='grey', alpha=.25)
    plt.legend(loc='best')
    plt.plot()
    
    errore = forecasted - valid
    errore.dropna(inplace=True)
    
    mse = mean_squared_error(valid, forecasted)
    print('MSE: %f' % mse)
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))
    
    #%%
    
    #EXPONENTIAL SMOOTHING
    
    # Questa parte serve per vedere se può essere moltiplicativo.
    # Però si eseguono divisioni con 0 quindi va messo: train[i] == 0
    # e train[i] = 1
    
    '''
    for i in range(1, len(train)):
        if train[i] == 0:
            train[i] = 1'''
    
    # Provare a cambiare i parametri per un migliore risultato...
    
    model = ExponentialSmoothing(train, trend="additive", damped = True, seasonal="additive", seasonal_periods=year)
    
    # fit model
    
    fitted = model.fit()
    
    # make prediction. Stesso periodo del validation set!
    
    model_predictions = fitted.forecast(steps = int(len(valid)))
    
    #aggiungo il risultato del simple exponential smoothing (perchè funziona meglio???)
    
    model_predictions = model_predictions + forecasted
    
    # tolgo i valori negativi
    
    for i in range(1, len(model_predictions)):
        if model_predictions[i] < 0:
            model_predictions[i] = 0
            
    for i in range(1, len(fitted.fittedvalues)):
        if fitted.fittedvalues[i] < 0:
            fitted.fittedvalues[i] = 0
            
    z = 1.96
    sse = fitted.sse
    for i in range(1, len(valid)):
        predint_xminus[i] = model_predictions[i] - z * np.sqrt(sse/len(valid)+i)
        predint_xplus[i]  = model_predictions[i] + z * np.sqrt(sse/len(valid)+i)
    
    #model_predictions = model_results.predict(start="2018-06-11", end="2019-09-29")
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(fitted.fittedvalues, label="Exponential Smoothing", color=MRC)
    plt.plot(model_predictions, label="Forecasts (in sample)", color=FC)
    plt.plot(predint_xminus, color="grey", alpha = .5)
    plt.plot(predint_xplus, color="grey", alpha = .5)
    plt.fill_between(pd.date_range(start="2018-06-11", periods=len(valid) , freq='D'), 
                 predint_xplus, 
                 predint_xminus, 
                 color='grey', alpha=.25)
    plt.legend(loc='best')
    plt.plot()
    
    errore = model_predictions - valid
    errore.dropna(inplace=True)
    
    mse = mean_squared_error(valid, model_predictions)
    print('MSE: %f' % mse)
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))

    
    # %%
    # Proviamo a usare ets in maniera scomposta sulle componenti e poi
    # sommare i risultati.
    #
    # NOTA: qua sommare i residual è di poco conto.
    
    decomposition = seasonal_decompose(train, period=year, two_sided=False)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    trend.dropna(inplace=True)
    seasonal.dropna(inplace=True)
    residual.dropna(inplace=True)

    # Creiamo dei modelli per trend e seasonal + USO ARIMA PER I RESIDUAL VISTO CHE SONO UNA COMPONENTE STAZION.
    trend_model = ExponentialSmoothing(trend, trend="add", damped = True, seasonal=None)
    seasonal_model = ExponentialSmoothing(seasonal, trend=None, seasonal='add', seasonal_periods=year)
    # ARIMA SU RESIDUAL (PER FORZA)
    residual_model = ARIMA(residual, order=(1, 0, 6))
    
    # fit model
    trend_fitted    = trend_model.fit()
    seasonal_fitted = seasonal_model.fit()
    residual_fitted = residual_model.fit()
    
    # make prediction. Stesso periodo del validation set!    
    trend_model_predictions = trend_fitted.forecast(steps = int(len(valid)))
    seasonal_model_predictions = seasonal_fitted.forecast(steps = int(len(valid)))
    residual_model_predictions, _, _ = residual_fitted.forecast(steps = int(len(valid)))

    #Sommo i modelli
    model_predictions = trend_model_predictions \
                        + seasonal_model_predictions \
                        + residual_model_predictions
                        
    model_predictions.dropna(inplace=True)
    
    # annulliamo i valori negativi    
    for i in range(1, len(model_predictions)):
        if model_predictions[i] < 0:
            model_predictions[i] = 0
        
    z = 1.96
    sse = trend_fitted.sse + seasonal_fitted.sse
    for i in range(1, len(model_predictions)):
        predint_xminus[i] = model_predictions[i] - z * np.sqrt(sse/len(valid)+i)
        predint_xplus[i]  = model_predictions[i] + z * np.sqrt(sse/len(valid)+i)
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(trend_fitted.fittedvalues + 
             seasonal_fitted.fittedvalues + 
             residual_fitted.fittedvalues, label="Exponential Smoothing", color=MRC)
    plt.plot(model_predictions, label="Forecasts (in sample)", color=FC)
    plt.plot(predint_xminus, color="grey", alpha = .5)
    plt.plot(predint_xplus, color="grey", alpha = .5)
    plt.fill_between(pd.date_range(start="2018-06-11", periods=len(valid) , freq='D'), 
                 predint_xplus, 
                 predint_xminus, 
                 color='grey', alpha=.25)
    plt.legend(loc='best')
    plt.plot()

    
    errore = model_predictions - valid
    errore.dropna(inplace=True)

    mse = mean_squared_error(valid, model_predictions)
    print('MSE: %f' % mse)
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))
    
