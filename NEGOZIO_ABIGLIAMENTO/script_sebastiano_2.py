# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:35:21 2020

@author: seba3

IMPORTANTE!!!!
Per installare tensorflow usare pip3!!!
!pip3 install tensorflow

- l'ultima volta mi sono dimenticato, per i grafici della rolling window in pag 24 
devi ridurre la dimensione della finestra che al momento sembra rispondere troppo lentamente 
ai cambi, scrivi inoltre esplicitamente che lunghezza stai usando per la finestra

-domanda: Il grafico di arima in pag 26 è sempre sulla serie differenziata a 365 a cui poi 
hai aggiunto i valori dell'anno prima giusto? perchè se non è così allora quelle predizioni 
sono chiaramente in-sample

- riguardo al discorso di sarimax a pag 27: ti sei ricordato di risommare il valore dell'anno 
prima? perchè io intendevo di applicare il sarimax alla serie differenziata con 365.
"""

import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mytools as mt
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools

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
    half_year = 183
    
    # STAGIONE 
    season = half_year
    
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
    
    # Costanti per spezzare la serie temporale sempre nello stesso punto
    
    END_TRAIN = ts.index[int(len(ts) * 0.8)]
    START_VALID = ts.index[int(len(ts)*0.8)+1]
        
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
    
    rolmean = ts.rolling(window=year).mean()
    rolstd = ts.rolling(window=year).std()
    
    # Plot della serie iniziale con rolling mean e rolling std (365 giorni e 182 giorni)
    
    # ANNO
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Serie Maglie: training set, validation set, moving average e std (finestra 365 giorni)')
    plt.ylabel('#Maglie vendute')
    plt.xlabel('Data')
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(rolmean, color=OLC, label='Rolling Mean',  linewidth=3)
    plt.plot(rolstd, color=OLC, label='Rolling Std', linestyle = '--',  linewidth=3)
    plt.legend(loc='best')
    plt.show(block=False)
    plt.plot()
    
    # META' ANNO
    
    rolmean = ts.rolling(window=half_year).mean()
    rolstd = ts.rolling(window=half_year).std()
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Serie Maglie: training set, validation set, moving average e std (finestra 182 giorni)')
    plt.ylabel('#Maglie vendute')
    plt.xlabel('Data')
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(rolmean, color=OLC, label='Rolling Mean',  linewidth=3)
    plt.plot(rolstd, color=OLC, label='Rolling Std', linestyle = '--',  linewidth=3)
    plt.legend(loc='best')
    plt.show(block=False)
    plt.plot()
    
    # SETTIMANA
    
    rolmean = ts.rolling(window=week).mean()
    rolstd = ts.rolling(window=week).std()
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Serie Maglie: training set, validation set, moving average e std (finestra 7 giorni)')
    plt.ylabel('#Maglie vendute')
    plt.xlabel('Data')
    plt.plot(train, label="training set", color=TSC)
    plt.plot(valid, label="validation set", color =VSC, linestyle = '--')
    plt.plot(rolmean, color=OLC, label='Rolling Mean',  linewidth=3)
    plt.plot(rolstd, color=OLC, label='Rolling Std', linestyle = '--',  linewidth=3)
    plt.legend(loc='best')
    plt.show(block=False)
    plt.plot()
    
    mt.ac_pac_function(train, lags = 400)
    
    #%%
    # Decompongo la serie
    # con periodo di 365 o 183 giorni (year e half_year)
    
    result = seasonal_decompose(train,  model = 'additive', period = season)

    #%%

    trend = result.trend
    seasonality = result.seasonal
    residuals = result.resid
    
    strength_seasonal = max(0, 1 - residuals.var()/(seasonality + residuals).var())
    print('La forza della stagionalità di periodo {} è: {}'.format(season, strength_seasonal))
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(trend)
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(residuals)
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(seasonality)
    
    trend.dropna(inplace = True)
    seasonality.dropna(inplace = True)
    residuals.dropna(inplace = True)
    
    mt.ac_pac_function(trend, lags = 50)
    mt.ac_pac_function(residuals, lags = 50)
    
    """
    p = q = range(0, 7)
    d = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    
    for param in pdq:
        try:
            mod = ARIMA(train, order=param)
            results = mod.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue
        
    model = ARIMA(train, order=(6, 0, 6))
    fitted = model.fit()
    
    #fitted.summary()
    
    predictions, _, confidence_int = fitted.forecast(steps = len(valid))
    ts_predictions = pd.Series(predictions, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')) 
    """
    #%%
    
    # genere le previsioni della componente stagionale usando il metodo seasonal naive
    
    predictions_seasonality = []
    for i in range (0, len(valid)):
        if i < season:
            predictions_seasonality.append(seasonality[len(seasonality)-season+i])
        else:
            predictions_seasonality.append(predictions_seasonality[i%season])
            
    # produca la serie temporale dalla lista di valori usando come indice le date del validation set        
            
    ts_predictions_seasonality = pd.Series(predictions_seasonality, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D'))
            
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(seasonality, label='stagionalità')
    plt.plot(ts_predictions_seasonality, label='previsione stagionalità')
    plt.legend(loc='best');
      
    #%%  
      
    model_trend = ARIMA(trend, order=(5, 0, 5))
    model_residuals = ARIMA(residuals, order=(3, 0, 3))
    
    fitted_trend = model_trend.fit(disp=0)
    fitted_residuals = model_residuals.fit(disp=0)
    
    # Torno alla forma iniziale sommando le componenti
    
    model = fitted_trend.fittedvalues + fitted_residuals.fittedvalues + seasonality
            
    # Calcolo le previsioni (per un periodo come valid, con cui fare il confronto
    # per determinarne la bontà)

    predictions_trend, _, _ = fitted_trend.forecast(steps = int(len(valid)))
    predictions_residuals, _, _ = fitted_residuals.forecast(steps = int(len(valid)))
    
    ts_predictions_trend = pd.Series(predictions_trend, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D'))
    ts_predictions_residuals = pd.Series(predictions_residuals, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D'))
    
    ts_predictions_trend.dropna(inplace=True) 
    ts_predictions_seasonality.dropna(inplace=True) 
    ts_predictions_residuals.dropna(inplace=True) 
    
    predictions = ts_predictions_residuals + ts_predictions_seasonality + ts_predictions_trend
    
    for i in range (0, len(model)):
        if model[i] < 0:
            model[i] = 0
    
    for i in range (0, len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
    
    # Plot del modello ARIMA con la serie per il training in scala originale
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label = "Training set", color = 'black')
    plt.plot(model, color='green', label='ARIMA')
    plt.plot(valid, color='black', linestyle='--')
    plt.title("Arima")
    plt.legend(loc='best');
    
    # Per ottenere le previsioni in scala originale devo aggiungere la componente stagionale
    # (settimanale) che posso ottenere con una media esponenziale o un "approccio naive" ossia
    # prendendo la stagionalità di k osservazioni passate
     
    ci = 1.96 * np.std(predictions)/np.mean(predictions)
    plt.plot(predictions, color="red", label='previsione con ARIMA')
    plt.title('Previsioni con ARIMA')
    plt.xlabel('Data')
    plt.ylabel('#Maglie vendute')
    plt.legend(loc='best')
    print(predictions.head())

#%%

    errore = predictions - valid
    errore.dropna(inplace=True)
    
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))
