# -*- coding: utf-8 -*-
"""
Scipt di forecasting dati di Albignasego
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import datetime 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller


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
    plt.plot(ts)
    plt.plot(ts,  label=label, linestyle=linestyle)
    plt.legend(loc='best');
    plt.show()

def plotAcf(ts, isPacf=False, both=False, lags=40):
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
    auto_cor = acf(ts, nlags=lags)
    part_auto_cor = pacf(ts, nlags=lags)
    
    plt.figure(figsize=(12,6))    
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
    auto_cor = acf(ts_diff, nlags=lags)
    part_auto_cor = pacf(ts_diff, nlags=lags)
    
    # Upper bound: 1.96/sqrt(T)
    bound = 1.96/((T)**(0.5))
    
    # Determiniamo p
    for i in range(0, len(part_auto_cor)):
        if part_auto_cor[i] <= bound:
            p = i
            break
    for i in range(0, len(auto_cor)):
        if part_auto_cor[i] <= bound:
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
                model = ARIMA(ts, order=(i, d, j)).fit(disp=-1)
                if (model.aic < min_aic):
                    p=i
                    q=j
                    min_aic=model.aic
                    result_model=model
            except:
                continue
    return (p,q, result_model)

def strength_seasonal_trend(ts):
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
    decomposition = seasonal_decompose(ts)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    strength_trend = max(0, 1 - residual.var()/(trend + residual).var())
    strength_seasonal = max(0, 1 - residual.var()/(seasonal + residual).var())
    
    return (strength_seasonal, strength_trend)

if __name__ == "__main__":
    
    dateparser = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d')
    dataframe = pd.read_csv('Dati_Albignasego/Whole period.csv', index_col = 0, date_parser=dateparser)
    
    # Analizziamo i dati graficamente
    for column in dataframe:
        # Recuperiamo una serie temporale
        serie_totale = dataframe[column]
        
        # training set dell'80% sui dati
        serie = serie_totale[pd.date_range(start=dataframe.index[0], 
                                        end=serie_totale.index[int(len(serie_totale) * 0.8)] , freq='D')]
        # Plottiamo la serie temporale
        timeplot(ts=serie, label=column)
        
        # Cerchiamo di capire se la serie è stagionale e/o presenta trend dalla funzione acf
        plotAcf(serie, both=True)
        
        # Forza delle componenti stagionalità e trend
        strength_s, strength_t = strength_seasonal_trend(serie)
        print('Forza dei gradi stagionalita (%s), trend(%s)'%(str(strength_s), str(strength_t)))
        
        # Dal grafico della funzione ACF osserviamo che serie presenta sia un trend
        # sia una stagionalità
        isStationary = test_stationarity(serie)
        
        if (isStationary):
            print('La serie temporale %s è stazionaria'%column)
        else:
            print('La serie temporale %s NON è stazionaria'%column)
        
        # Visto che la serie delle serie è stazionaria allora allora per ARIMA d=0
        # Determiniamo dunque il valore di p e q con le funzioni ACF e PACF su la serie differenziata
        # p = il valore di k dove la pacf interseca con 1.96/sqrt(T)
        # q = il valore di k dove la acf interseca con 1.96/sqrt(T)
        serie_diff=serie.diff()
        serie_diff.dropna(inplace=True)
        plotAcf(serie_diff, both=True)
        
        # Definiamo dunque il modello ARIMA    
        p,q = p_QForArima(serie_diff, len(serie))
        
        model = ARIMA(serie, order=(p, int(not isStationary), q))
        results_arima = model.fit(disp=-1)
        
        # Ricordo che essendo la serie stazionaria non devo fare le somme cumulative
        arima_model = pd.Series(results_arima.fittedvalues, copy=True)
        
        # Plottamo grafico normale e arima_model
        plt.figure(figsize=(40, 20), dpi=80)
        plt.plot(serie, color='black', label=column)
        plt.plot(arima_model, color='red', label='Modello ARIMA(' + str(p) + ',' + str(int(not isStationary)) + ',' + str(q) +')')
        plt.legend(loc='best');
        plt.show()
        
        # Individuiamo il miglior modello ARIMA che approssima meglio la serie temporale
        best_p, best_q, best_result_model = find_best_model(serie, d=int(not isStationary))
        
        # Mettiamo in confronto i due modelli
        plt.figure(figsize=(40, 20), dpi=80)
        plt.subplot(211)
        plt.plot(serie, label=column, color='black')
        plt.plot(arima_model,color='red', label='Modello ARIMA(' + str(p) + ',' + str(int(not isStationary)) + ',' + str(q) +')')
        plt.legend(loc='best');
        plt.subplot(212)
        plt.plot(serie, label=column, color='black')
        best_arima_model = pd.Series(best_result_model.fittedvalues, copy=True)
        plt.plot(best_arima_model, color='green', label='Modello ARIMA(' + str(best_p) + ',' + str(int(not isStationary)) + ',' + str(best_q) +')')
        plt.legend(loc='best');
        plt.show()
        
        
        # Usiamo il miglior modello per fare le previsioni
        
        h = 50  # orizzonte
        
        
        previsione, _ ,intervallo = best_result_model.forecast(steps=h)
        
        plt.figure(figsize=(40, 20), dpi=80)
        plt.plot(best_arima_model, color="green", label='Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
        plt.plot(pd.date_range(start=serie.index[len(serie) - 1], periods=h , freq='D'), 
                 previsione, linestyle='-',color='red', label='Previsioni', alpha=.75)

        plt.plot(serie_totale[pd.date_range(
            start=serie_totale.index[int(len(serie_totale) * 0.8)], 
            periods=h , freq='D')], linestyle='-', 
            color='black', label='Osservazioni reali', markersize=7)
        
        intervallo_sup = [0.0] * len(intervallo)
        intervallo_inf = [0.0] * len(intervallo)
        
        # Normalizzo gli array
        ind = 0
        for n in intervallo[:, [0]]:
            intervallo_sup[ind] = float(n)
            ind+=1
        ind = 0
        for n in intervallo[:, [1]]:
            intervallo_inf[ind] = float(n)
            ind+=1
        
        plt.fill_between(pd.date_range(start=serie.index[len(serie) - 1], periods=h , freq='D'), 
                         intervallo_sup, 
                         intervallo_inf, 
                         color='black', alpha=.25)
        plt.legend(loc='best');
        plt.show()