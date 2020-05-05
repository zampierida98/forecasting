# coding: utf-8
"""
Created on Wed Apr 22 14:46:48 2020

@author: seba3
"""
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def test_stationarity(timeseries, temporalwindow, boolprint, position=0):
    
    """
    Utilizza il metodo Dickey-Fuller per ricavare i dati sulla stazionarietà
    della serie.
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)\n
        temporalwindow -> la finestra temporale per calcolare media e dev std in movimento (int)\n
        boolprint -> true per stampare il grafico con media e dev std, false se non mi interessa stampare (bool)\n
        position -> intero riga/colonna/cella, max 9 celle. Si può omettere per generare una nuova figure\n
    """
    if(boolprint):
    #Determina media e deviazione standard (rolling)
        rolmean = timeseries.rolling(window=temporalwindow).mean()
        rolstd = timeseries.rolling(window=temporalwindow).std()
    # Plot rolling statistiche (media e deviazione standard (sqrt di varianza) in movimento):
        if(position != 0):
            plt.subplot(position)
        else:
            plt.figure(figsize=(40, 20), dpi=80)
        plt.plot(timeseries, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # Valori critici
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    print('\n')
    
def make_seasonal_stationary(timeseries, temporalwindow, boolprint):
    
    """
    Rende stazionaria una serie temporale in cui è presente seasonality (indicata da temporal window).
    -----------------
    Metodo:
    -----------------
        differenza moving average
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)\n
        temporalwindow -> la finestra temporale per calcolare media in movimento (int)\n
        boolprint -> true per stampare i grafici, false se non mi interessa stampare (bool)\n
        
    """
    moving_avg = timeseries.rolling(window = temporalwindow).mean()
    #I primi 12 mesi presenteranno valore Nan in quanto non esiste una media da sottrarre...
    timeseries_moving_avg_diff = timeseries - moving_avg
    #...per questo motivo li cancello
    timeseries_moving_avg_diff.dropna(inplace=True)
    
    if(boolprint):
        plt.figure(figsize=(40, 20), dpi=80)
        plt.plot(timeseries, color='blue', label='ts logaritmica')
        plt.plot(timeseries, color='red', label='moving average della ts logaritmica')
        plt.plot(timeseries_moving_avg_diff, color='green', label='ts logaritmica - moving average')
        plt.show(block=False)
        
    return timeseries_moving_avg_diff

def make_exponential_stationary(timeseries, decayfactor, boolprint):
    
    """
    Rende stazionaria una serie temporale in cui non è possibile individuare stagionalità nei dati.
    -----------------
    Metodo:
    -----------------
        differenza moving average exponentially weighted con peso assegnato ai valori precedenti che
        diminuisce in base alla distanza dal valore attuale
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)\n
        boolprint -> true per stampare i grafici, false se non mi interessa stampare (bool)\n
        decayfactor -> valore per indicare una stima dell'exponential decay'\n
    """

    expwighted_avg = timeseries.ewm(decayfactor).mean()
    ts_log_ewma_diff = timeseries - expwighted_avg
    if(boolprint):
        plt.plot(timeseries, color='blue', label='ts_logaritmica')
        plt.plot(expwighted_avg, color='red', label='exponentially weighted moving average')
        plt.plot(ts_log_ewma_diff, color='green', label='ts logaritmica - ewma')
        plt.show(block=False)
    return ts_log_ewma_diff

def log_transform(timeseries):
    
    """
    Trasformazione logaritmica della serie originale. Non applicare se ci sono valori nulli nella serie!
    -----------------
    Parametri
    -----------------
        timeseries -> la serie temporale (dataframe)
    """
    return np.log(timeseries)

def differencing(timeseries):
    
    """
    Applica una differenziazione
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)
    """
    differenced_ts = timeseries - timeseries.shift()
    ts = differenced_ts.dropna(inplace=True)
    return ts

def decompose(timeseries):
    
    """
    Scompone la serie temporale in trend, season e residual e stampa a video.
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)
    """
    
    decomposition = seasonal_decompose(timeseries)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.subplot(411)
    plt.plot(timeseries, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show(block=False)
    
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    return ts_log_decompose

def ac_pac_function(timeseries, pos1=0, pos2=0):
    
    """
    Calcola le funzioni di autocorrelazione e autocorrelazione parziale di una serie temporale.
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale resa stazionaria con un metodo qualsiasi (dataframe)\n
        pos1 -> [OPZIONALE] posizione primo subplot (intero riga/colonna/cella max 9 celle)\n
        pos2 -> [OPZIONALE] posizione secondo subplot (intero riga/colonna/cella max 9 celle)\n
    """
    
    lag_acf = acf(timeseries, nlags=20)
    lag_pacf = pacf(timeseries, nlags=20, method='ols')
    
    if pos1==0 and pos2==0:
        plt.figure(figsize=(40, 20), dpi=80)
        #Plot ACF: 
        plt.subplot(211) 
    else:
        plt.subplot(pos1)
        
    plt.plot(lag_acf)
    #Delimito i tre intervalli
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='red')
    plt.title('Autocorrelation Function')
    
    if pos1==0 and pos2==0:
        #Plot PACF:
        plt.subplot(212)
    else:
        plt.subplot(pos2)
   
    plt.plot(lag_pacf)
    #Delimito i tre intervalli
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='red')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='red')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    