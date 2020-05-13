# -*- coding: utf-8 -*-
"""
Scipt di forecasting dati di Albignasego
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import datetime 
#import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller

SMALL_SIZE = 28 
MEDIUM_SIZE = 30
BIGGER_SIZE = 32
COLOR_ORIG = 'black'
COLOR_MODEL = 'green'
COLOR_FOREC = 'red'
COLOR_ACF = 'blue'
 
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def timeplot(ts, label, linestyle='-'):
    '''
    Funzione che fa realizza un plot di una serie temporale
    
    Parameters
    ----------
    ts : TYPE
        pandas.Series
    label : TYPE, optional
        Label del plot. The default is ''.
    linestyle : TYPE, optional
        Stile della linea del plot. The default is '-'.

    Returns 
    -------
    None.
    '''
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(label)
    plt.plot(ts)
    plt.plot(ts, label=label, linestyle=linestyle, color=COLOR_ORIG)
    plt.legend(loc='best');
    plt.show()

def plotAcf(ts, isPacf=False, both=False, lags=40, title=""):
    '''
    Funzione che realizza un plot della acf/pacf (o entrambe) 
    data una serie temporale
    Parameters
    ----------
    ts : pandas.Series
        Serie temporale
    isPacf : bool
        Se vogliamo invece di acf la pacf
    both : bool
        Se vogliamo entrambi acf e pacf in un unico grafico
    lags : int
        Quanti valori deve assumere k di autocor o p_autocor
    Returns
    -------
    None.
    '''
    auto_cor = acf(ts, nlags=lags, fft=True)
    part_auto_cor = pacf(ts, nlags=lags)
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(title)
    if not both:
        if not isPacf:
            plt.plot(auto_cor)
        else:
            plt.plot(part_auto_cor)
        plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    else:
        plt.subplot(211)
        plt.plot(auto_cor,  label='ACF')
        plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--',color='red')
        plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--',color='red')
        plt.legend(loc='best');
        plt.subplot(212)
        plt.plot(part_auto_cor, label='PACF')
        plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
        plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.legend(loc='best');
    plt.show()
    
def test_stationarity(ts):
    '''
    https://www.performancetrading.it/Documents/PmEconometria/PmEq_Dickey_Fuller.htm
    Esegue un test di stazionarietà usando il test Dickey-Fuller. 
    Controlliamo se il valore di 'test statistic' 
    è minore del valore critico 1% così da avere una buona certezza di stazionarietà
    Parameters
    ----------
    ts : pandas.Series
        Serie temporale

    Returns
    -------
    True se stazionaria, False non stazionaria
    
    '''
    
    print('Risultati del test di Dickey-Fuller aumentato:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    # Valori critici
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput);print('\n')
    
    return dftest[0] < dftest[4]['10%']     # Abbiamo il 90% che la serie sia stazionaria

def p_QForArima(ts_diff, T, lags=40):
    '''
    È la funzione che determina a partire dalla funzione acf e pacf i valori di
    p e q senza guardare i grafici.
    Parameters
    ----------
    ts_diff : pandas.Series
        Serie temporale differenziata
    T : int
        Numero di osservazioni della serie 
    lags : int, optional
        Quanti valori deve assumere k di autocor e p_autocor. The default is 40.

    Returns
    -------
    p : int
        La p di arima
    q : int
        la q di arima

    '''
    p = 0
    q = 0
    auto_cor = acf(ts_diff, nlags=lags, fft=True)
    part_auto_cor = pacf(ts_diff, nlags=lags)
    
    # Upper bound: 1.96/sqrt(T)
    bound = 1.96/((T)**(0.5))
    
    # Determiniamo p
    for i in range(0, len(part_auto_cor)):
        if part_auto_cor[i] <= bound:
            p = i
            break
    for i in range(0, len(auto_cor)):
        if auto_cor[i] <= bound:
            q = i
            break
    return (p,q)


def find_best_model(ts, d=0, max_p=5, max_q=5):
    '''
    È la funzione che cerca il miglior modello ARMA per una serie temporale.
    Il miglior modello è colui che minimizza il grado AIC
    Parameters
    ----------
    ts : pandas.Series
        Serie temporale
    d : int, optional
        d di ARIMA(p,d,q). The default is 0.
    max_p : int, optional
        max valore di cerca per p. The default is 5.
    max_q : int, optional
        max valore di cerca per q. The default is 5.
    Returns 
    -------
    (p,d, result_model) : (int, int, ARIMAResults)
    '''
    min_aic = 2**32; p = 0; q = 0
    result_model = None
    for i in range(0, max_p):
        for j in range(0, max_q):
            try:
                print('order(%d,%d,%d)'%(i,d,j))
                model = ARIMA(ts, order=(i, d, j)).fit(disp=-1)
                if (model.aic < min_aic):
                    p=i
                    q=j
                    min_aic=model.aic
                    result_model=model
            except:
                continue
    return (p,q, result_model)

def strength_seasonal_trend(ts, season=12):
    '''
    La funzione calcola, sfruttando la decomposizione delle ts, la forza della
    stagionalità e del trend
    
    Parameters
    ----------
    ts : pandas.Series
        Serie temporale

    Returns
    -------
    strength_seasonal : float
        Grado della stagionalità
    strength_seasonal_trend : float
        Grado del trend

    '''
    decomposition = seasonal_decompose(ts, period=season)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    strength_trend = max(0, 1 - residual.var()/(trend + residual).var())
    strength_seasonal = max(0, 1 - residual.var()/(seasonal + residual).var())
    
    return (strength_seasonal, strength_trend)

if __name__ == "__main__":
    len_lag = 150
    season = 365
    dateparser = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d')
    dataframe = pd.read_csv('./Dati_Albignasego/Whole period.csv', index_col = 0, date_parser=dateparser)
    
    # Analizziamo i dati graficamente
    for column in {'MAGLIE'}:#dataframe:
        # Recuperiamo una serie temporale
        serie_totale = dataframe[column]
        # tiriamo fuori il training set dell'80% sui dati
        serie = serie_totale[pd.date_range(start=dataframe.index[0], end=serie_totale.index[int(len(serie_totale) * 0.8)], freq='D')]
        
        # Eliminiamo il 29 Febbraio al fine di recuperare poi la componenete stagionale 
        serie = serie.drop(labels=[pd.Timestamp('2016-02-29')])
        
        # Plottiamo la serie temporale
        timeplot(ts=serie, label=column)
        
        # Cerchiamo di capire se la serie è stagionale e/o presenta trend dalla funzione acf
        plotAcf(serie, both=True, title="Acf " + column) #, lags=int(len(serie)/3)
        
        # Forza delle componenti stagionalità e trend
        strength_s, strength_t = strength_seasonal_trend(serie, season)
        print('Forza dei gradi stagionalita (%s), trend(%s)'%(str(strength_s), str(strength_t)))
        
        # Dal grafico della funzione ACF osserviamo che serie presenta stagionalità
        isStationary = test_stationarity(serie)
        
        if (isStationary):
            print('La serie temporale %s è stazionaria'%column)
        else:
            print('La serie temporale %s NON è stazionaria'%column)

        # Rimuovo la componente stagionale per semplificarmi la vita nella previsione 
        decomposition = seasonal_decompose(serie, period=season)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title("Decomposizione serie")
        plt.subplot(411)
        plt.plot(serie, label='Original', color=COLOR_ORIG)
        plt.legend(loc='best');
        plt.subplot(412)
        plt.plot(trend, label='Trend', color=COLOR_ACF)
        plt.legend(loc='best');
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal', color=COLOR_ACF)
        plt.legend(loc='best');
        plt.subplot(414)
        plt.plot(residual, label='Residuals', color=COLOR_ACF)
        plt.legend(loc='best');
        plt.show()
        
        # Serie de stagionata
        serie_destagionata = trend + residual
        serie_destagionata.dropna(inplace=True)
        
        # Realizzazione ARIMA        
        serie_diff = serie_destagionata.diff()
        serie_diff.dropna(inplace=True)
        plotAcf(serie_diff, both=True, lags=100, title="Acf e Pacf della serie " + column + " differenziata")
        
        # Definiamo dunque il modello ARIMA    
        p,q = p_QForArima(serie_diff, len(serie_diff))        
        model = ARIMA(serie_destagionata, order=(p, int(not isStationary), q))
        results_arima = model.fit(disp=-1)
        arima_model = pd.Series(results_arima.fittedvalues, copy=True)
        
        # Plottamo grafico normale e arima_model
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title("Serie destagionata:" + column + ", ARIMA(" + str(p) + ',' + str(int(not isStationary)) + ',' + str(q) +')')
        plt.plot(serie_destagionata, label=column, color=COLOR_ORIG)
        plt.plot(arima_model, label='Modello ARIMA(' + str(p) + ',' + str(int(not isStationary)) + ',' + str(q) +')', color=COLOR_MODEL)
        plt.legend(loc='best');
        plt.show()        
        
        # Individuiamo il miglior modello ARIMA che approssima meglio la serie_destagionata temporale
        best_p, best_q, best_result_model = find_best_model(serie_destagionata, d=int(not isStationary))
        
        # %%
        # Mettiamo in confronto i due modelli
        plt.figure(figsize=(40, 20), dpi=80)
        plt.plot("CONFRONTO FRA: " + 'Modello ARIMA(' + str(p) + ',' + str(0) + ',' + str(q) +')' + " e " + 'Modello ARIMA(' + str(best_p) + ',' + str(0) + ',' + str(best_q) +')')
        plt.subplot(211)
        plt.plot(serie_destagionata, label=column, color=COLOR_ORIG)
        plt.plot(arima_model, label='Modello ARIMA(' + str(p) + ',' + str(0) + ',' + str(q) +')', color=COLOR_MODEL)
        plt.legend(loc='best');
        plt.subplot(212)
        plt.plot(serie_destagionata, label=column, color=COLOR_ORIG)
        
        best_arima_model = pd.Series(best_result_model.fittedvalues, copy=True)
        plt.plot(best_arima_model, label='Modello ARIMA(' + str(best_p) + ',' + str(0) + ',' + str(best_q) +')', color=COLOR_MODEL)
        plt.legend(loc='best');
        plt.show()
        
        # %%
        # Uniamo al miglior arima trovato la componente stagionale
        best_arima_model_con_stag = best_arima_model + seasonal
        best_arima_model_con_stag.dropna(inplace=True)
        
        '''
        plt.figure(figsize=(40, 20), dpi=80)
        plt.plot(serie[(best_arima_model_con_stag).index], label=column, color='black')
        plt.plot(best_arima_model_con_stag,color='red', label='Modello ARIMA(' + str(p) + ',' + str(0) + ',' + str(q) +')')
        plt.legend(loc='best');
        plt.show()
        '''
        
        # DEVO RICORDARMI UNO SFASAMENTO A CAUSA DEL FATTO CHE USO INTERI COME INDICI        
        sfasamento = int((len(seasonal) - len(best_arima_model_con_stag))/2)   
        
        # FORECASTING
        h = len(serie_totale) - len(seasonal)  # orizzonte        
        last_observation = best_arima_model_con_stag.index[len(best_arima_model_con_stag) - 1]        
        
        ts_seasonal_forecast = pd.Series(seasonal[best_arima_model_con_stag.index], copy=True)
        #ts_seasonal_forecast = ts_seasonal_forecast.add(seasonal[pd.date_range(start=last_observation, periods=h , freq='D')], fill_value=0)
        
        # Previsione sulla parte stagionale usando la parte stagionale "periodica"
        ts_seasonal_forecast = pd.Series(seasonal[best_arima_model_con_stag.index], copy=True)
        
        # Devo calcolare le previsioni sulla componente stagionale
        tmp = [0.0] * h                         # conterrà i valori di previsione stagionale, dati dalla media dei valori dello stesso periodo
        start = len(ts_seasonal_forecast)       # rappresenta l'osservazione futura da prevedere
        
        for i in range(0, h): # 0 sarebbe t+1 e arriva a t+1+h-1=t+h
            ind = start
            #tmp[i] = seasonal[i - season]  # seasonal naif
            
            alpha = 0.9 # sommatoria in media exp
            ind -= season # prima il decremento perchè non abbiamo il valore di t+1 
            tmp[i] += seasonal[sfasamento + ind]
            exp = 1
            while (ind >= 0):
                tmp[i] += seasonal[sfasamento + ind] * ((1 - alpha) ** exp)
                exp += 1
                ind -= season # prima il decremento perchè non abbiamo il valore di t+1 
            
            start += 1 # questo arriverà fino a t+h
            tmp[i] = tmp[i]
            
        
        ts_seasonal_forecast_h = pd.Series(data=tmp, index=pd.date_range(start=last_observation, periods=h , freq='D'))
        ts_seasonal_forecast = ts_seasonal_forecast.add(ts_seasonal_forecast_h, fill_value=0)#seasonal[pd.date_range
        
        # ADESSO DEVO AGGIUNGERE UNA PARTE DI PREVISIONE SU SEASONAL_FORECAST
        # CHE PARTE DAL SEASONAL E ARRIVA FINO ALLA PREVISIONE TOTALE SU SERIE_TOTALE
        
        tmp = [0.0] * sfasamento                         # conterrà i valori di previsione stagionale, dati dalla media dei valori dello stesso periodo
        start = len(ts_seasonal_forecast)       # rappresenta l'osservazione futura da prevedere
        
        for i in range(0, sfasamento): # 0 sarebbe t+1 e arriva a t+1+h-1=t+h
            ind = start
            #tmp[i] = seasonal[i - season]  # seasonal naif
            
            alpha = 0.9 # sommatoria in media exp
            ind -= season # prima il decremento perchè non abbiamo il valore di t+1 
            tmp[i] += ts_seasonal_forecast[ind]
            exp = 1
            while (ind >= 0):
                tmp[i] += ts_seasonal_forecast[ind] * ((1 - alpha) ** exp)
                exp += 1
                ind -= season # prima il decremento perchè non abbiamo il valore di t+1 
            
            start += 1 # questo arriverà fino a t+h
            tmp[i] = tmp[i]
            
        
        ts_seasonal_forecast_h = pd.Series(data=tmp, index=pd.date_range(start=ts_seasonal_forecast.index[len(ts_seasonal_forecast) - 1], periods=sfasamento, freq='D'))
        ts_seasonal_forecast = ts_seasonal_forecast.add(ts_seasonal_forecast_h, fill_value=0)
        
        
        # H con sfasamento
        new_h = h + sfasamento
        
        # Previsioni sulla parte de-stagionata
        previsione, _ ,intervallo = best_result_model.forecast(steps=new_h)
        
        ts_NOseasonal_forecast = pd.Series(previsione, index=pd.date_range(start=last_observation, periods=new_h, freq='D'))
        
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title("PREVISIONI CON " + 'Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
        plt.plot(best_arima_model_con_stag, color=COLOR_MODEL, label='Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
        plt.plot(ts_seasonal_forecast + ts_NOseasonal_forecast,color=COLOR_FOREC, label='Previsioni')

        plt.plot(serie_totale[pd.date_range(
            start=last_observation, 
            periods=new_h, freq='D')], linestyle='-', 
            color=COLOR_ORIG, label='Osservazioni reali')
        
        intervallo_sup = [0.0] * new_h
        intervallo_inf = [0.0] * new_h
        seasonal_interval_sum = [0.0] * new_h
        
        # Normalizzo gli array
        ind = 0
        for n in intervallo[:, [0]]:
            intervallo_sup[ind] = float(n)
            ind+=1
        ind = 0
        for n in intervallo[:, [1]]:
            intervallo_inf[ind] = float(n)
            ind+=1

        # Recupero i valori di ts_seasonal_forecast
        ind = 0
        for i in range(len(ts_seasonal_forecast) - new_h, len(ts_seasonal_forecast)):
            seasonal_interval_sum[ind] = float(ts_seasonal_forecast[i])
            ind+=1

        # SOMMATORIA
        for i in range(0, new_h):
            intervallo_sup[i] += seasonal_interval_sum[i]
        for i in range(0, new_h):
            intervallo_inf[i] += seasonal_interval_sum[i]
        
        plt.fill_between(pd.date_range(start=last_observation, periods=new_h , freq='D'), 
                         intervallo_sup, 
                         intervallo_inf, 
                         color=COLOR_ORIG, alpha=.25)
        plt.legend(loc='best');
        plt.show()