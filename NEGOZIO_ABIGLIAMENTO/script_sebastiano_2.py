# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:35:21 2020

@author: seba3

IMPORTANTE!!!!
Per installare tensorflow usare pip3!!!
!pip3 install tensorflow
"""

import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mytools as mt
import datetime as dt
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
    half_year = 182
    
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
    
    # Traccio i grafici ACF e PACF per evidenziare come sia presente una stagionalità
    # con periodo settimanale
    
    result = seasonal_decompose(ts)
    mt.ac_pac_function(result.seasonal, lags = 50)
    mt.ac_pac_function(ts, lags = 50)
    
    #%%
    
    # Calcolo la serie differenziata con finestra di 7 giorni e ne traccio il grafico
    # per vedere se c'è stato un miglioramento nella stazionarietà della serie
    
    my_ts = train.diff(periods=7)  
    my_ts.dropna(inplace = True)
    
    rolmean = my_ts.rolling(window=week).mean()
    rolstd = my_ts.rolling(window=week).std()
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Moving average e std del train set differenziato')
    plt.ylabel('#Maglie vendute')
    plt.xlabel('Data')
    plt.plot(my_ts, label="training set", color=TSC)
    plt.plot(rolmean, color=OLC, label='Rolling Mean',  linewidth=2)
    plt.plot(rolstd, color=OLC, label='Rolling Std', linestyle = '--',  linewidth=2)
    plt.legend(loc='best')
    plt.show(block=False)
    plt.plot()
    
    # Traccio il grafico delle funzioni di correlazione e autocorrelazione parziale per
    # estrarre i pesi p e q da usare nel modello arima
    
    mt.ac_pac_function(my_ts, lags = 100)
    p, q = 4, 6

    #%%

    # calcolo la componente stagionale che mi servirà per tornare nella forma iniziale

    my_stagionalita = train.rolling(window=7).mean()
    my_stagionalita.dropna(inplace= True)
    
    model = ARIMA(my_ts, order=(p, 0, q))
    results_ARIMA = model.fit(disp=0)
    
    # Torno alla forma iniziale aggiungendo la componente stagionale
    
    my_arima = pd.Series(results_ARIMA.fittedvalues, copy=True)
    original_scale = my_arima + my_stagionalita
    
    # sistemo alcuni "errori del modello". Livello a 0 tutto ciò che scende
    # sotto (non ci possono essere "vendite negative")
    
    for i in range(1, len(original_scale)):
        if original_scale[i] < 0:
            original_scale[i] = 0
            
    # Calcolo le previsioni (per un periodo come valid, con cui fare il confronto
    # per determinarne la bontà)

    predictions, _, interval = results_ARIMA.forecast(steps = int(len(valid)))
    predictions = pd.Series(predictions, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D'))
    predictions.dropna(inplace=True) 
    
    # Plot del modello ARIMA con la serie per il training differenziata
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(my_ts, label = "Training set", color = 'black')
    plt.plot(my_arima, color='green', label='ARIMA')
    plt.plot(predictions, color='red', label='previsioni')
    plt.title("Arima (%d, 0, %d)"% (p, q))
    plt.legend(loc='best');
    
    # Plot del modello ARIMA con la serie per il training in scala originale
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label = "Training set", color = 'black')
    plt.plot(original_scale, color='green', label='ARIMA')
    plt.title("Arima (%d, 0, %d)"% (p, q))
    plt.legend(loc='best');
    
    # Traccio il grafico delle previsioni in scala originale
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label='training set', color = 'black')
    plt.plot(valid, color='black', label='validation set', linestyle = '--')
    plt.plot(original_scale, color = 'green', label = 'risultati di ARIMA')
    
    #%%
    
    # Per ottenere le previsioni in scala originale devo aggiungere la componente stagionale
    # (settimanale) che posso ottenere con una media esponenziale o un "approccio naive" ossia
    # prendendo la stagionalità di k osservazioni passate
    
    """forecast = predictions+ts.rolling(window=week).mean()"""
    
    seasonal_factor = pd.Series.copy(valid)
    for i in range(0, len(valid)):
        seasonal_factor[i] = train[len(train)-1]
        
    seasonal_factor.dropna(inplace=True)
        
    forecast =pd.Series.copy(predictions)    
    for i in range(0, len(valid)):
        forecast[i] += seasonal_factor[i]
                                       
    forecast.dropna(inplace = True)

    for i in range(1, len(forecast)):
        if forecast[i] < 0:
            forecast[i] = 0
     
    ci = 1.96 * np.std(forecast)/np.mean(forecast)
    plt.plot(forecast, color="red", label='previsione con ARIMA')
    plt.title('Previsioni con ARIMA('+str(p)+',0,'+str(q)+')')
    plt.xlabel('Data')
    plt.ylabel('#Maglie vendute')
    plt.legend(loc='best')
    print(predictions.head())

#%%

    errore = forecast - valid
    errore.dropna(inplace=True)
    
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))

    