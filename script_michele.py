# -*- coding: utf-8 -*-
"""
Scipt di forecasting dati di Albignasego
"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

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
    Funzione che realizza un plot della funzione acf/pacf (o entrambe) 
    di una serie temporale
    Parameters
    ----------
    ts : TYPE
        DESCRIPTION.
    isPacf : TYPE
        DESCRIPTION.

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
    è minore del valore critico 1%
    Parameters
    ----------
    ts : TYPE
        Serie temporale

    Returns True se stazionaria, False non stazionaria
    -------
    '''
    
    print('Risultati del test di Dickey-Fuller aumentato:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    # Valori critici
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput);print('\n')
    
    return dftest[0] < dftest[4]['1%']

def p_QForArima(ts_diff, T, lags=40):
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
    best_model = None
    for i in range(0, max_p):
        for j in range(0, max_q):
            model = ARIMA(ts, order=(i, d, j)).fit(disp=-1)
            if (model.aic < min_aic):
                p=i
                q=j
                min_aic=model.aic
                result_model=model

    return (p,q, result_model)
if __name__ == "__main__":
    # COLONNE #,MAGLIE,CAMICIE,GONNE,PANTALONI,VESTITI,GIACCHE
    
    #Whole period.csv
    dateparser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    dataframe = pd.read_csv('Dati_Albignasego/Whole period.csv',index_col = 0, date_parser=dateparser)
    
    # Analizziamo i dati graficamente
    '''
    for column in dataframe:
        timeplot(ts=dataframe[column], label=column)'''    
    # Commentato perchè impiega davvero tanto tempo
    
    # Lavoriamo intanto con le MAGLIE
    maglie = dataframe['MAGLIE']
    
    # Commento il training set dell'80% sui dati
    '''
    maglie = maglie[pd.date_range(start=dataframe.index[0], 
                                    end=maglie.index[int(len(maglie) * 0.8)] , freq='D')]
    '''
   
    # Cerchiamo di capire se la serie è stagionale e/o presenta trend dalla funzione acf
    plotAcf(maglie, both=True)
    
    # Dal grafico della funzione ACF osserviamo che maglie presenta sia un trend
    # sia una stagionalità
    
    if (test_stationarity(maglie)):
        print('La serie temporale %s è stazionaria'%'MAGLIE')
    else:
        print('La serie temporale %s NON è stazionaria'%'MAGLIE')
    
    # Visto che la serie delle MAGLIE è stazionaria allora allora per ARIMA d=0
    # Determiniamo dunque il valore di p e q con le funzioni ACF e PACF su la serie differenziata
    # p = il valore di k dove la pacf interseca con 1.96/sqrt(T)
    # q = il valore di k dove la acf interseca con 1.96/sqrt(T)
    maglie_diff=maglie.diff()
    maglie_diff.dropna(inplace=True)
    plotAcf(maglie_diff, both=True)
    
    # Definiamo dunque il modello ARIMA    
    p,q = p_QForArima(maglie_diff, len(maglie))
    
    model = ARIMA(maglie, order=(p, 0, q))
    results_arima = model.fit(disp=-1)
    
    # Ricordo che essendo la serie stazionaria non devo fare le somme cumulative
    arima_model = pd.Series(results_arima.fittedvalues, copy=True)
    
    # Plottamo grafico normale e arima_model
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(maglie, color='black', label='MAGLIE')
    plt.plot(arima_model, color='red', label='Modello ARIMA(' + str(p) + ',0,' + str(q) +')')
    plt.show()
    
    # Individuiamo il miglior modello ARIMA che approssima meglio la serie temporale
    best_p, best_q, best_result_model = find_best_model(maglie)
    #best_p, best_q, best_result_model = (4,4,ARIMA(maglie, order=(4, 0, 4)).fit(disp=-1))
    
    # Mettiamo in confronto i due modelli
    plt.figure(figsize=(40, 20), dpi=80)
    plt.subplot(211)
    plt.plot(maglie, label='MAGLIE', color='black')
    plt.plot(arima_model,color='red', label='Modello ARIMA(' + str(p) + ',0,' + str(q) +')')
    plt.legend(loc='best');
    plt.subplot(212)
    plt.plot(maglie, label='MAGLIE', color='black')
    best_arima_model = pd.Series(best_result_model.fittedvalues, copy=True)
    plt.plot(best_arima_model, color='green', label='Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
    plt.legend(loc='best');
    plt.show()
    
    
    # Usiamo il miglior modello per fare le previsioni
    
    h = 50  # orizzonte 
    
    previsione, _ ,intervallo = best_result_model.forecast(steps=h)
    plt.figure(figsize=(40, 20), dpi=80)
    plt.plot(best_arima_model, color="green", label='Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
    plt.plot(pd.date_range(start=maglie.index[len(maglie) - 1], periods=h , freq='D'), 
             previsione, linestyle='--',color='red', label='Previsioni')
    plt.plot(pd.date_range(start=maglie.index[len(maglie) - 1], periods=h , freq='D'), 
             intervallo, linestyle='--', color='red')
    plt.legend(loc='best');
    plt.show()
    