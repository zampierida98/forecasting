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
    
    #%%
    # Traccio i grafici ACF e PACF per evidenziare come sia presente una stagionalità
    # con periodo settimanale
    
    result = seasonal_decompose(train,  model = 'additive', period = year)
    
    mt.ac_pac_function(result.seasonal, lags = 40)
    mt.ac_pac_function(ts, lags = 40)
    mt.ac_pac_function(result.seasonal, lags = 100)
    mt.ac_pac_function(ts, lags = 100)
    
    # Traccio il grafico delle funzioni di correlazione e autocorrelazione parziale per
    # estrarre i pesi p e q da usare nel modello arima
    
    mt.ac_pac_function(train, lags = 100)

    #%%

    trend = result.trend
    seasonality = result.seasonal
    residuals = result.resid
    
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
    mt.ac_pac_function(seasonality, lags = 50)
    
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
    """
        
    """
    ARIMA(0, 0, 0) - AIC:15812.293282757695
    ARIMA(0, 0, 1) - AIC:15110.175017936563
    ARIMA(0, 0, 2) - AIC:14976.100385185735
    ARIMA(0, 0, 3) - AIC:14902.075594298336
    ARIMA(0, 0, 4) - AIC:14818.340425486223
    ARIMA(0, 0, 5) - AIC:14818.493394782914
    ARIMA(0, 0, 6) - AIC:14782.926407316168
    ARIMA(0, 1, 0) - AIC:15194.933774743744
    ARIMA(0, 1, 1) - AIC:14725.377087929124
    ARIMA(0, 1, 2) - AIC:14573.081743822604
    ARIMA(0, 1, 3) - AIC:14572.28801608449
    ARIMA(0, 1, 4) - AIC:14529.217299006177
    ARIMA(0, 1, 5) - AIC:14524.533751990484
    ARIMA(0, 1, 6) - AIC:14495.503409426154
    ARIMA(1, 0, 0) - AIC:14826.876641323724
    ARIMA(1, 0, 1) - AIC:14710.802715996979
    ARIMA(1, 0, 2) - AIC:14564.11660577477
    ARIMA(1, 0, 3) - AIC:14562.789396719543
    ARIMA(1, 0, 4) - AIC:14511.268794083848
    ARIMA(1, 0, 5) - AIC:14502.313150259186
    ARIMA(1, 0, 6) - AIC:14463.682148810423
    ARIMA(1, 1, 0) - AIC:15072.845865002868
    ARIMA(1, 1, 1) - AIC:14609.346722413444
    ARIMA(1, 1, 2) - AIC:14573.46794587272
    ARIMA(1, 1, 3) - AIC:14538.90396563551
    ARIMA(1, 1, 4) - AIC:14520.926139638463
    ARIMA(1, 1, 5) - AIC:14522.768742094047
    ARIMA(1, 1, 6) - AIC:14488.702784481613
    ARIMA(2, 0, 0) - AIC:14814.727963779227
    ARIMA(2, 0, 1) - AIC:14600.326244572903
    ARIMA(2, 0, 2) - AIC:14564.241370858474
    ARIMA(2, 0, 3) - AIC:14515.045056685789
    ARIMA(2, 0, 4) - AIC:14495.203493317003
    ARIMA(2, 0, 5) - AIC:14496.76390975958
    ARIMA(2, 0, 6) - AIC:14449.25376935316
    ARIMA(2, 1, 0) - AIC:14883.728140971936
    ARIMA(2, 1, 1) - AIC:14556.39295676736
    ARIMA(2, 1, 2) - AIC:14486.298297077336
    ARIMA(2, 1, 3) - AIC:14269.105834457096
    ARIMA(2, 1, 4) - AIC:14388.497293762091
    ARIMA(2, 1, 5) - AIC:14210.115906164023
    ARIMA(2, 1, 6) - AIC:14208.172174558258
    ARIMA(3, 0, 0) - AIC:14735.67465648305
    ARIMA(3, 0, 1) - AIC:14546.424114798589
    ARIMA(3, 0, 2) - AIC:14457.898280334977
    ARIMA(3, 0, 3) - AIC:14535.603775969017
        HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
    ARIMA(3, 0, 4) - AIC:14378.650199866272
    ARIMA(3, 0, 5) - AIC:14189.33367244909
    ARIMA(3, 0, 6) - AIC:14189.99108798486
    ARIMA(3, 1, 0) - AIC:14815.37535035214
    ARIMA(3, 1, 1) - AIC:14550.74514601448
    ARIMA(3, 1, 2) - AIC:14474.620357987773
    ARIMA(3, 1, 3) - AIC:14394.552521761228
    ARIMA(3, 1, 4) - AIC:14245.82142372298
    ARIMA(3, 1, 5) - AIC:14210.989615817596
    ARIMA(3, 1, 6) - AIC:14204.416696136508
    ARIMA(4, 0, 0) - AIC:14710.00081633976
    ARIMA(4, 0, 1) - AIC:14540.416820930493
    ARIMA(4, 0, 2) - AIC:14445.146760640448
    ARIMA(4, 0, 3) - AIC:14385.32268738848
    ARIMA(4, 0, 4) - AIC:14374.890583584542
    ARIMA(4, 0, 5) - AIC:14190.90205976865
    ARIMA(4, 0, 6) - AIC:14192.315541040422
    ARIMA(4, 1, 0) - AIC:14761.435824666602
    ARIMA(4, 1, 1) - AIC:14504.058145111116
    ARIMA(4, 1, 2) - AIC:14286.126951191638
    ARIMA(4, 1, 3) - AIC:14176.945063197609
    ARIMA(4, 1, 4) - AIC:14176.068302926751
    ARIMA(4, 1, 5) - AIC:13985.05312971261
    ARIMA(4, 1, 6) - AIC:13986.98418457412
    ARIMA(5, 0, 0) - AIC:14685.229385975847
    ARIMA(5, 0, 1) - AIC:14493.419631600073
    ARIMA(5, 0, 2) - AIC:14237.699893829078
    ARIMA(5, 0, 3) - AIC:14158.46088278529
    ARIMA(5, 0, 4) - AIC:14153.82429377347
    ARIMA(5, 0, 5) - AIC:13970.003193852966
    ARIMA(5, 0, 6) - AIC:13971.388702652825
    ARIMA(5, 1, 0) - AIC:14475.257945875694
    ARIMA(5, 1, 1) - AIC:14325.083222038578
    ARIMA(5, 1, 2) - AIC:14146.886032872542
    ARIMA(5, 1, 3) - AIC:14148.69182646366
    ARIMA(5, 1, 4) - AIC:14127.056295571168
    ARIMA(5, 1, 5) - AIC:13986.987401023232
    ARIMA(5, 1, 6) - AIC:13984.432500223456
    ARIMA(6, 0, 0) - AIC:14444.713695793676
    ARIMA(6, 0, 1) - AIC:14314.882105193448
    ARIMA(6, 0, 2) - AIC:14123.450053333312
    ARIMA(6, 0, 3) - AIC:14124.12489016474
    ARIMA(6, 0, 4) - AIC:14079.26022800937
    ARIMA(6, 0, 5) - AIC:13971.443897019653
    ARIMA(6, 0, 6) - AIC:13969.760128446998 <--- scelgo questo
    ARIMA(6, 1, 0) - AIC:14176.270673755089
    ARIMA(6, 1, 1) - AIC:14170.659250033319
    ARIMA(6, 1, 2) - AIC:14158.819364118975
    ARIMA(6, 1, 3) - AIC:14133.280103030385
    ARIMA(6, 1, 4) - AIC:14052.196075133104
    ARIMA(6, 1, 5) - AIC:13988.659787201963
    ARIMA(6, 1, 6) - AIC:13985.037311157492
    """
    """
    model = ARIMA(train, order=(6, 0, 6))
    fitted = model.fit()
    
    #fitted.summary()
    
    predictions, _, confidence_int = fitted.forecast(steps = len(valid))
    ts_predictions = pd.Series(predictions, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D')) 
    """
    #%%
    
    predictions_seasonality = []
    for i in range (0, len(valid)):
        if i < 365:
            predictions_seasonality.append(seasonality[len(seasonality)-365+i])
        else:
            predictions_seasonality.append(predictions_seasonality[i%365])
      
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
    ts_predictions_seasonality = pd.Series(predictions_seasonality, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D'))
    ts_predictions_residuals = pd.Series(predictions_residuals, index=pd.date_range(start=ts.index[int(len(ts)*0.8)+1], end = ts.index[int(len(ts))-1], freq='D'))
    
    ts_predictions_trend.dropna(inplace=True) 
    ts_predictions_seasonality.dropna(inplace=True) 
    ts_predictions_residuals.dropna(inplace=True) 
    
    predictions = ts_predictions_residuals + ts_predictions_seasonality + ts_predictions_trend
    
    # Plot del modello ARIMA con la serie per il training in scala originale
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(train, label = "Training set", color = 'black')
    plt.plot(fitted.fittedvalues, color='green', label='ARIMA')
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

    errore = forecast - valid
    errore.dropna(inplace=True)
    
    print("Calcoliamo  MAE=%.4f"%(sum(abs(errore))/len(errore)))

#%%

    plt.plot(seasonality)
    plt.plot(ts_predictions_seasonality)