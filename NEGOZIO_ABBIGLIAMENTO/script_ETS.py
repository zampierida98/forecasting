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
    
    # Settiamo i colori 
    
    TSC = 'black'   # training set
    VSC = 'black'   # validation set
    FC = 'red'      # previsioni
    MRC = 'green'   # model results
    OLC = 'orange'  # other lines
    
    # Specifichiamo due stagionalita'
    year = 365 # giorni
    week = 7
    
    # L'insieme di dati contiene la data e il numero di capi di abbigliamento venduti
    # in quel giorno (per tipo).
    # Carichiamo dunque i dati in un oggetto DataFrame
    
    dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
    data = pd.read_csv('./Dati_Albignasego/Whole period.csv', index_col=0, date_parser=dateparse)
    
    # Usiamo solo la serie maglie. Il procedimento si può ripetere con ciascun capo...
    
    ts = data['MAGLIE'] 
        
    # Se si vuole togliere il 29 febbraio 2016 per avere solo anni di 365 giorni. 
    # Sconsigliato se si considera una stagionalità settimanale in quanto sfalsa di un giorno.
    """
    ts = ts.drop(labels=[pd.Timestamp('2016-02-29')])
    print(ts_totale['2016-02'])
    """
    
    # Definiamo il training set (80% dei dati) e il validation set (20% dei dati)
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
    #SIMPLE EXPONENTIAL SMOOTHING...
    
    # Creiamo il modello usando la classe SimpleExpSmoothing
    modelv1 = SimpleExpSmoothing(train)
    
    # Facciamo l'adattamento del modello ai dati
    fitted = modelv1.fit()
    
    # Creiamo le previsioni sullo stesso periodo del validation set
    forecasted = fitted.predict(start="2018-06-11", end="2019-09-29")
    
    # Calcolo gli intervalli di predizione per Simple Exponential Smoothing.
    # Sommiamo/Sottraiamo alle previsioni all'istante t-esimo il valore di 
    # c*sigma dove questo valore rappresenta il 95% della guassiana.
    
    predint_xminus = ts[pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')]
    predint_xplus  = ts[pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')]
    
    z = 1.96
    sse = fitted.sse
    for i in range(1, len(valid)):
        predint_xminus[i] = forecasted[i] - z * np.sqrt(sse/len(valid)+i)
        predint_xplus[i]  = forecasted[i] + z * np.sqrt(sse/len(valid)+i)
    
    # Andiamo a graficare il risultato e vediamo che è poco soddisfacente.
    # Del resto sappiamo che Simple Exp Smoothing crea previsioni costanti
    # ovvero y_{t+h} = y_{t+1} dove y_{t+1} è la previsione usando un set
    # di valori lungo t.

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
    
    # Calcoliamo la variabile d'errore 
    errore = forecasted - valid
    errore.dropna(inplace=True)
    
    # Calcoliamo le metriche d'errore
    mse = mean_squared_error(valid, forecasted)
    print('MSE: %f' % mse)
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))
    
    #%%
    
    #EXPONENTIAL SMOOTHING
    
    # Creiamo il modello con trend Additivo e crescita non lineare quindi
    # damped = True (previsioni più verosimili; nel tempo tendono a diventari quasi
    # costanti). Stagionalità additiva perchè plottando i grafici della serie temporale
    # delle maglie abbiamo visto che la stagionalità non aumenta/diminuisce all'aumentare
    # del livello della serie
    model = ExponentialSmoothing(train, trend="additive", damped = True, seasonal="additive", seasonal_periods=year)
    
    # Adattiamo il modello ai dati
    fitted = model.fit()
    
    # Creiamo le previsioni lunghe quanto il validation set
    model_predictions = fitted.forecast(steps = int(len(valid)))
    
    # Ci siamo accorti dai residui che questi non erano a media nulla ma avevano
    # un valore pari al valore del Simple Exponential Smoothing. Quindi abbiamo
    # sommato quella costante moltiplicativa alle previsioni ottenendo dei risultati
    # migliori
    
    model_predictions = model_predictions + forecasted
    
    # togliamo i valori negativi dalle previsioni perchè non possono esserci vendite
    # di capi negative
    for i in range(1, len(model_predictions)):
        if model_predictions[i] < 0:
            model_predictions[i] = 0
            
    for i in range(1, len(fitted.fittedvalues)):
        if fitted.fittedvalues[i] < 0:
            fitted.fittedvalues[i] = 0
    
    # creiamo gli intervalli di predizione.
    # Sommiamo/Sottraiamo alle previsioni all'istante t-esimo il valore di 
    # c*sigma dove questo valore rappresenta il 95% della guassiana.
    z = 1.96
    sse = fitted.sse
    for i in range(1, len(valid)):
        predint_xminus[i] = model_predictions[i] - z * np.sqrt(sse/len(valid)+i)
        predint_xplus[i]  = model_predictions[i] + z * np.sqrt(sse/len(valid)+i)
    
    #model_predictions = model_results.predict(start="2018-06-11", end="2019-09-29")
    
    # Plottiamo i grafici e vediamo i risultati
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
    
    # Calcoliamo la variabile d'errore
    errore = model_predictions - valid
    errore.dropna(inplace=True)
    
    # Calcoliamo le metriche d'errore
    mse = mean_squared_error(valid, model_predictions)
    print('MSE: %f' % mse)
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))
    
    # %%
    # Proviamo a usare ETS applicandolo alle componenti trend e stagionalità.
    # Per i residui, essendo una serie di rumore bianco (priva di componenti),
    # viene usato ARIMA perchè con ETS potremmo al limite usare Simple Exponential Smoothing
    # ma non riesce a generare previsioni soddisfacenti.
    #
    # NOTA: qua sommare i residual è di poco conto.
    
    # Decomponiamo la serie temporale. 
    # two_sided=False significa che la media mobile (processo descritto nel notebook)
    # viene calcolata a partire dai valori passati invece che essere normalmente centrata.
    decomposition = seasonal_decompose(train, period=year, two_sided=False)
    
    # Recuperiamo le componenti
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Rimuoviamo eventuali valori NaN dalle serie
    trend.dropna(inplace=True)
    seasonal.dropna(inplace=True)
    residual.dropna(inplace=True)

    # Creiamo dei modelli per trend e seasonal
    # USO ARIMA PER I RESIDUAL VISTO CHE SONO UNA COMPONENTE STAZIONARIA
    
    trend_model = ExponentialSmoothing(trend, trend="add", damped = True, seasonal=None)
    seasonal_model = ExponentialSmoothing(seasonal, trend=None, seasonal='add', seasonal_periods=year)
    
    # ARIMA SU RESIDUAL
    residual_model = ARIMA(residual, order=(1, 0, 6))
    
    # Facciamo il fitting dei modelli sui dati
    trend_fitted    = trend_model.fit()
    seasonal_fitted = seasonal_model.fit()
    residual_fitted = residual_model.fit()
    
    # Creiamo le previsioni con lo stesso periodo del validation set
    trend_model_predictions = trend_fitted.forecast(steps = int(len(valid)))
    seasonal_model_predictions = seasonal_fitted.forecast(steps = int(len(valid)))
    residual_model_predictions, _, _ = residual_fitted.forecast(steps = int(len(valid)))

    # Sommiamo i modelli
    model_predictions = trend_model_predictions \
                        + seasonal_model_predictions \
                        + residual_model_predictions
    
    # Rimuoviamo alcuni valori NaN
    model_predictions.dropna(inplace=True)
    
    # Annulliamo i valori negativi    
    for i in range(1, len(model_predictions)):
        if model_predictions[i] < 0:
            model_predictions[i] = 0
        
    # Calcoliamo gli intervalli di previsioni usando ormai il consueto modo
    
    z = 1.96
    sse = trend_fitted.sse + seasonal_fitted.sse
    for i in range(1, len(model_predictions)):
        predint_xminus[i] = model_predictions[i] - z * np.sqrt(sse/len(valid)+i)
        predint_xplus[i]  = model_predictions[i] + z * np.sqrt(sse/len(valid)+i)
    
    # Rappresentiamo i grafici
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

    # Calcoliamo l'errore
    errore = model_predictions - valid
    errore.dropna(inplace=True)

    # Calcoliamo le metriche d'errore
    mse = mean_squared_error(valid, model_predictions)
    print('MSE: %f' % mse)
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('/home/ayhan/international-airline-passengers.csv', 
                 parse_dates=['Month'], 
                 index_col='Month'
)
df.index.freq = 'MS'
train, test = df.iloc[:130, 0], df.iloc[130:, 0]
model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12).fit()
pred = model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='best')
"""