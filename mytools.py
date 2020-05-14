# coding: utf-8
"""
Created on Wed Apr 22 14:46:48 2020

@author: seba3
"""
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def kpss_test(timeseries):
    
    """
    Test per determinare la stazionarietà del trend.
    -----------------
    Parametri:
    -----------------
         timeseries -> la serie temporale (dataframe)\n
    """
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    if(kpss_output['Test Statistic']<kpss_output['Critical Value (1%)']):
        print("La serie è trend stazionaria (test kpss)")
    else:
        print("La serie non è trend stazionaria (test kpss)")

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
        position -> [OPZIONALE] intero riga/colonna/cella, max 9 celle. Si può omettere per generare una nuova figure\n
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
        plt.plot(timeseries, label='Original')
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
    if(dfoutput['Test Statistic']<dfoutput['Critical Value (1%)']):
        print("La serie è stazionaria (test Dickey-Fuller)")
    else:
        print("La serie non è stazionaria (test Dickey-Fuller)")
    
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
        plt.plot(timeseries, label='ts logaritmica')
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
        plt.plot(timeseries, label='ts_logaritmica')
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
        timeseries -> la serie temporale (dataframe)\n
    """
    timeseries = np.log(timeseries)
    return timeseries

def differencing(timeseries, period=0):
    
    """
    Applica una differenziazione (lag 1 oppure stagionale, a seconda del periodo passato)
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)\n
        period -> lag dello shift da applicare\n
    """
    if period == 0:
        timeseries = timeseries.diff()
    else:
        timeseries = timeseries - timeseries.shift(period)
    timeseries.dropna(inplace=True)
    return timeseries

def cumulative_sums(ts_diff, season, ts_iniz):
    
    """
    Restituisce la serie pre-differenziazione. Funziona anche cancellando il 29 di Febbraio.
    -----------------
    Parametri:
    -----------------
        ts_diff -> la serie temporale differenziata (dataframe)\n
        season -> l'offset temporale applicato per differenziare la serie
        ts_iniz -> la serie temporale che ha subito la differenziazione
    """

    restored = ts_iniz.copy()
    restored.iloc[season:] = np.nan
    counter = 0
    for d, val in ts_diff.iloc[season:].iteritems():
        restored[d] = restored.iloc[counter-season]+val
        counter+=1
    
    return restored
      

def decompose(timeseries):
    
    """
    Scompone la serie temporale in trend, season e residual e stampa a video.
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale (dataframe)\n
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
    
    ts_decompose = residual
    ts_decompose.dropna(inplace=True)
    return ts_decompose

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
    
def p_q_for_ARIMA(timeseries):
    
    """
    Calcola gli ordini p e q per il modello ARIMA (operazione eseguibile a vista, qui è stata resa
    automatica)
    -----------------
    Parametri:
    -----------------
        timeseries -> la serie temporale resa stazionaria con un metodo qualsiasi (dataframe)\n
    """
    ACF = acf(timeseries, nlags=20)
    PACF = pacf(timeseries, nlags=20)
    limite = 1.96/np.sqrt(len(timeseries))
    p = 0
    q = 0
   
    for i in range(0, len(PACF)):
        if PACF[i] <= limite:
            p = i
            break
    for i in range(0, len(ACF)):
        if ACF[i] <= limite:
            q = i
            break
    
    return (p,q)
def forza_season(timeseries, season=12):
    """
    Calcola il grado di forza di una presunta stagione nella serie
    -----------------
    Parametri
    -----------------
    timeseries -> la serie temporale 
    season -> la presunta lunghezza della stagione di cui misurare il "grado di forza"
    """
    decomposition = seasonal_decompose(timeseries, period=season)
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    forza_stagionale = max(0, 1 - residual.var()/(seasonal + residual).var())
    
    return forza_stagionale
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    