# -*- coding: utf-8 -*-
"""
Scipt di forecasting dati di Albignasego
"""

# IMPORTIAMO LE DIVERSE LIBRERIE
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller

# SOPPRIMIAMO I WARNING DI NUMPY
np.warnings.filterwarnings('ignore')

# SETTIAMO DIVERSI PARAMETRI GRAFICI
SMALL_SIZE = 32
MEDIUM_SIZE = 34
BIGGER_SIZE = 40
COLOR_ORIG = 'black'
COLOR_MODEL = 'green'
COLOR_FOREC = 'red'
COLOR_ACF = 'blue'
 
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# DEFINISCO LE DIVERSE FUNZIONI

def timeplot(ts, label, linestyle='-'):
    '''
    Funzione che fa realizza un plot di una serie temporale
    
    Parameters
    ----------
    ts :  pandas.Series
        Serie temporale
    label : string
        Label del plot. The default is ''.
    linestyle : string, optional
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
    # Calcoliamo le funzioni di autocorrelazione e di autocorrelazione parziale
    auto_cor = acf(ts, nlags=lags, fft=True)
    part_auto_cor = pacf(ts, nlags=lags)
    
    # Creiamo il grafico
    plt.figure(figsize=(40, 20), dpi=80)
    
    # se vogliamo o ACF o PACF
    if not both:
        if not isPacf:
            plt.plot(auto_cor)
            plt.title("ACF di " + title)
        else:
            plt.plot(part_auto_cor)
            plt.title("PACF di " + title)
        
        # Settiamo le righe che identificano valori non approssimabili come 0
        plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    
    # Se li vogliamo entrambi
    else:
        plt.subplot(211)
        plt.title("ACF di " + title)
        plt.plot(auto_cor,  label='ACF')
        
        # Settiamo le righe che identificano valori non approssimabili come 0
        plt.axhline(y=-1.96/np.sqrt(len(ts)), linestyle='--',color='red')
        plt.axhline(y=1.96/np.sqrt(len(ts)), linestyle='--',color='red')
        plt.legend(loc='best');
        
        plt.subplot(212)
        plt.title("PACF di " + title)
        plt.plot(part_auto_cor, label='PACF')
        
        # Settiamo le righe che identificano valori non approssimabili come 0
        plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
        plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.legend(loc='best');
    plt.show()
    
def test_stationarity(ts):
    '''
    https://www.performancetrading.it/Documents/PmEconometria/PmEq_Dickey_Fuller.htm
    Esegue un test di stazionarietà usando il test Dickey-Fuller aumentato. 
    Controlliamo se il valore di 'test statistic'
    è minore del valore critico 10% così da avere una buona certezza di stazionarietà
    
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
    
    # Cambiando la percentuale e impiegandola più piccola avremmo un maggiore grado di certezza
    
    return dftest[0] < dftest[4]['10%']     # Abbiamo il 90% che la serie sia stazionaria


def p_QForArima(ts_diff, T, lags=40):
    '''
    È la funzione che determina a partire dalla funzione acf e pacf i valori di
    p e q usando l'approccio classico di identificazioni dei parametri di ARIMA.
    
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
    
    # Calcoliamo le funzioni di ACF e PACF della serie differenziata
    auto_cor = acf(ts_diff, nlags=lags, fft=True)
    part_auto_cor = pacf(ts_diff, nlags=lags)
    
    # Settiamo le righe che identificano valori non approssimabili come 0
    bound = 1.96/((T)**(0.5))
    
    # Determiniamo come p il valore k tale che PACF[k] è minore di bound
    for i in range(0, len(part_auto_cor)):
        if part_auto_cor[i] <= bound:
            p = i
            break
        
    # Determiniamo come p il valore k tale che ACF[k] è minore di bound
    for i in range(0, len(auto_cor)):
        if auto_cor[i] <= bound:
            q = i
            break
    return (p,q)


def find_best_model(ts, d=0, max_p=5, max_q=5):
    '''
    È la funzione che cerca il miglior modello ARIMA per una serie temporale.
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
    
    
    min_aic = 2**32     # Settimo un valore molto alto in modo che la funzione min selezioni il primo valore AIC del primo modello
    p = 0
    q = 0
    result_model = None
    
    for i in range(0, max_p):
        for j in range(0, max_q):
            
            # try non tutti i modelli arima si adattano ai dati
            try:
                # Stampa dell'ordine cosi' l'utente non si innervosisce
                print('order(%d,%d,%d)'%(i,d,j))
                
                # Definiamo il modello e eseguiamo il fit dei dati.
                # Il fit determina i migliori coefficiente usando tecniche
                # simili alla minimizzazione dei residui quadratici
                
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
    La funzione calcola, sfruttando il principio di decomposizione, la forza della
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
    # Decomponiamo la serie temporale
    decomposition = seasonal_decompose(ts, period=season)
    
    # Estraiamo le componenti elementari
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Calcoliamo la forza mettendo in rapporto la varianza dei residui con la
    # Serie de-stagionalizzata (per la forza del trend) o la serie de-trendizzata (per la forza stagionale)
    
    strength_trend = max(0, 1 - residual.var()/(trend + residual).var())
    strength_seasonal = max(0, 1 - residual.var()/(seasonal + residual).var())
    
    return (strength_seasonal, strength_trend)

if __name__ == "__main__":
    # Defiamo la variabile che indica la stagionalità della serie
    season = 365
    
    # Carichiamo il file Whole period.csv in un oggetto DataFrame
    dateparser = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d')
    dataframe = pd.read_csv('./Dati_Albignasego/Whole period.csv', index_col = 0, date_parser=dateparser)
    
    for column in {'MAGLIE'}: #dataframe:
        
        # Recuperiamo la serie temporale dal DATAFRAME
        serie_totale = dataframe[column]
        
        # tiriamo fuori il training set dell'80% sui dati
        serie = serie_totale[pd.date_range(start=pd.Timestamp('2013-03-23'), end=serie_totale.index[int(len(serie_totale) * 0.8)], freq='D')]
        
        # Eliminiamo il 29 Febbraio per avere anni da 365 giorni
        serie = serie.drop(labels=[pd.Timestamp('2016-02-29')])
        
        # Plottiamo la serie temporale
        timeplot(ts=serie, label=column)
        
        # Cerchiamo di capire se la serie è stagionale e/o presenta trend dalla funzione acf
        plotAcf(serie, both=True, title=column) #, lags=int(len(serie)/3)
        
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
        
        # Realizziamo dei plot con tutte le componenti separate
        plt.figure(figsize=(40, 35), dpi=80)
        plt.subplot(411)
        plt.plot(serie, color=COLOR_ORIG)
        plt.title("Serie originale di " + column)
        plt.legend(loc='best');
        plt.subplot(412)
        plt.plot(trend, color=COLOR_ACF)
        plt.title("Trend di " + column)
        plt.legend(loc='best');
        plt.subplot(413)
        plt.title("Stagionalità di " + column)
        plt.plot(seasonal, color=COLOR_ACF)
        plt.legend(loc='best');
        plt.subplot(414)
        plt.title("Residui di " + column)
        plt.plot(residual, color=COLOR_ACF)
        plt.legend(loc='best');
        plt.show()
        
        
        # Calcoliamo la media mobile e anche la dev. standard mobile.
        # Cosi' capiamo se la media e/o la dev.standand è costante o meno
        # nel tempo
        
        rolmean = residual.rolling(window=15).mean()
        rolstd = residual.rolling(window=15).std()

        # Realizziamo i plot per le rolling mean
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title('Studio residui con rolling window su media e dev. standard di %s'%column)
        plt.plot(residual, label='Residui', color = 'black')
        plt.plot(rolmean, color='orange', label='Rolling Mean',  linewidth=3)
        plt.plot(rolstd, color='orange', label='Rolling Std', linestyle = '--',  linewidth=3)
        plt.legend(loc='best');
        plt.show()
        
        # %%
        # Defiamo la serie de-stagionalizzata
        serie_destagionata = trend + residual
        serie_destagionata.dropna(inplace=True)
        
        # PREVISIONI CON ARIMA
        # 
        # 
        # 
        
        # Calcoliamo la serie de-stagionalizzata differenziata e realizziamo
        # le funzioni ACF e PACF per capire graficamente i valori di P e Q
        serie_diff = serie_destagionata.diff()
        serie_diff.dropna(inplace=True)
        plotAcf(serie_diff, both=True, lags=10, title=column + " differenziata")
        
        # Definiamo ora il modello ARIMA usando i coefficienti trovati con le
        # funzioni ACF e PACF
        
        # Calcoliamo i coefficienti
        p,q = p_QForArima(serie_diff, len(serie_diff))        
        
        # Calcoliamo il modello e poi facciamo il fitting dei dati
        model = ARIMA(serie_destagionata, order=(p, int(not isStationary), q))
        results_arima = model.fit(disp=-1)
        
        # Creiamo di ARIMA un oggetto pandas.Series
        arima_model = pd.Series(results_arima.fittedvalues, copy=True)
        
        # Grafichiamo la serie de-stagionalizzata insieme al modello ARIMA
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title("Serie destagionata:" + column + ", ARIMA(" + str(p) + ',' + str(int(not isStationary)) + ',' + str(q) +')')
        plt.plot(serie_destagionata, label=column, color=COLOR_ORIG)
        plt.plot(arima_model, label='Modello ARIMA(' + str(p) + ',' + str(int(not isStationary)) + ',' + str(q) +')', color=COLOR_MODEL)
        plt.legend(loc='best');
        plt.show()        
        
        
        # Individuiamo il miglior modello ARIMA che approssima meglio la serie_destagionata
        best_p, best_q, best_result_model = find_best_model(serie_destagionata, d=int(not isStationary))
        
        # Creiamo di best_result_model un oggetto pandas.Series
        best_arima_model = pd.Series(best_result_model.fittedvalues, copy=True)
        
        # Mettiamo in confronto i due modelli ARIMA trovati. Quello dove i coefficienti vengono
        # calcolati con le funzioni ACF e PACF e ARIMA trovato minimizzando il valore AIC
        plt.figure(figsize=(40, 25), dpi=80)
        plt.subplot(211)
        plt.title('Modello ARIMA(' + str(p) + ',' + str(0) + ',' + str(q) +') di ' + column)
        plt.plot(serie_destagionata, label=column, color=COLOR_ORIG)
        plt.plot(arima_model, label='Modello ARIMA(' + str(p) + ',' + str(0) + ',' + str(q) +')', color=COLOR_MODEL)
        plt.legend(loc='best');
        plt.subplot(212)
        plt.title('MIGLIOR MODELLO: Modello ARIMA(' + str(best_p) + ',' + str(0) + ',' + str(best_q) +') di ' + column)
        plt.plot(serie_destagionata, label=column, color=COLOR_ORIG)
        plt.plot(best_arima_model, label='Modello ARIMA(' + str(best_p) + ',' + str(0) + ',' + str(best_q) +')', color="steelblue")
        plt.legend(loc='best');
        plt.show()
        
        # %%
        # DOBBIAMO CALCOLARE LE PREVISIONI DELLA COMPONENTE STAGIONALE
        # E QUESTO VIENE FATTO "MANUALMENTE"
        
        # Uniamo al miglior arima trovato la componente stagionale
        best_arima_model_con_stag = best_arima_model + seasonal
        best_arima_model_con_stag.dropna(inplace=True)
    
        # DEVO RICORDARMI UNO SFASAMENTO A CAUSA DEL FATTO CHE USO INTERI COME INDICI        
        sfasamento = int((len(seasonal) - len(best_arima_model_con_stag))/2)   
        
        # FORECASTING
        # 
        # 
        
        # Definiamo h orizzonte di previsione. Rappresenta il salto nel futuro
        # in numero di giorni
        h = len(serie_totale) - len(seasonal)  # orizzonte     
        
        # Determino l'ultima osservazione del miglior modello
        last_observation = best_arima_model_con_stag.index[len(best_arima_model_con_stag) - 1]        
        
        # Defiamo la serie temporale che conterra' i valori delle previsioni della componente stagionale
        ts_seasonal_forecast = pd.Series(seasonal[best_arima_model_con_stag.index], copy=True)
        

        # Iniziamo a calcolare le previsioni sulla componente stagionale
        # Calcolo il valore medio all’istante t come media dei valori t 
        # delle stagioni precedenti.
        # In realtà, non ho realizzato una normale media ma ho optato per una media exp (do maggior peso ai valori vicini)
        # 
        
        tmp = [0.0] * h                    # conterrà i valori di previsione stagionale, dati dalla media dei valori dello stesso periodo
        start = len(ts_seasonal_forecast)  # rappresenta la prima osservazione futura da prevedere d
        
        for i in range(0, h): # 0 sarebbe t+1 e arriva a t+1+h-1=t+h
            ind = start
            
            alpha = 0.9     # sommatoria in media exp
            ind -= season   # prima il decremento perchè non abbiamo il valore di t+1 
            tmp[i] += seasonal[sfasamento + ind]
            exp = 1
            
            while (ind >= 0):
                tmp[i] += seasonal[sfasamento + ind] * ((1 - alpha) ** exp)
                exp += 1
                ind -= season # prima il decremento perchè non abbiamo il valore di t+1 
            
            start += 1 # questo arriverà fino a t+h
            tmp[i] = tmp[i]
            
        
        # Creiamo delle previsioni sulla serie stagionale un oggetto pandas.Series 
        ts_seasonal_forecast_h = pd.Series(data=tmp, index=pd.date_range(start=last_observation, periods=h , freq='D'))
        ts_seasonal_forecast = ts_seasonal_forecast.add(ts_seasonal_forecast_h, fill_value=0)#seasonal[pd.date_range
        
        
        # ADESSO DEVO AGGIUNGERE UNA PARTE DI PREVISIONE A TS_SEASONAL_FORECAST
        # IN MODO DA ARRIVARE FINO ALLA STESSA LUNGHEZZA DI SERIE_TOTALE
        
        tmp = [0.0] * sfasamento          # conterrà i valori di previsione stagionale, dati dalla media dei valori dello stesso periodo
        start = len(ts_seasonal_forecast) # rappresenta l'osservazione futura da prevedere
        
        for i in range(0, sfasamento): # 0 sarebbe t+1 e arriva a t+1+h-1=t+h
            ind = start
            
            alpha = 0.9   # sommatoria in media exp
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
        
        # ORIZZONTE CHE TIENE CONTO DELLO SFASAMENTO
        new_h = h + sfasamento
        
        # Previsioni sulla parte de-stagionata
        previsione, _ ,intervallo = best_result_model.forecast(steps=new_h)
        
        # Le previsioni appena calcolate vengono tradotte in pandas.Series
        ts_NOseasonal_forecast = pd.Series(previsione, index=pd.date_range(start=last_observation, periods=new_h, freq='D'))
        
        # Iniziamo a rappresentare i modelli arima
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title("Previsioni di " + column + " con il " + 'Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
        plt.plot(serie_totale, color=COLOR_ORIG, label='Osservazioni reali')
        plt.plot(best_arima_model_con_stag, color=COLOR_MODEL, label='Modello ARIMA(' + str(best_p) + ',0,' + str(best_q) +')')
        
        # SOMMIAMO le previsioni della serie stagionale e della serie de-stagionalizzata
        ts_forecast = ts_seasonal_forecast + ts_NOseasonal_forecast
        plt.plot(ts_forecast,color=COLOR_FOREC, label='Previsioni')
        
        # Calcoliamo gli intervalli di previsione 
        # definendo l'intervallo superiore e inferiore a partire dalla variabile
        # "intervallo" ottenuta dopo aver usato il metodo forecast 
        
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

        # Recupero i valori di ts_seasonal_forecast in modo da sommarli agli intervalli
        # superiori e inferiori

        ind = 0
        for i in range(len(ts_seasonal_forecast) - new_h, len(ts_seasonal_forecast)):
            seasonal_interval_sum[ind] = float(ts_seasonal_forecast[i])
            ind+=1

        # Sommiamo i valori di seasonal_interval_sum agli intervalli superiori ed inferiori

        for i in range(0, new_h):
            intervallo_sup[i] += seasonal_interval_sum[i]
        for i in range(0, new_h):
            intervallo_inf[i] += seasonal_interval_sum[i]

        # Rappresentiamo in grigio l'intervallo di previsione a partire dalle
        # due liste sopra definiti
        plt.fill_between(pd.date_range(start=last_observation, periods=new_h , freq='D'), 
                         intervallo_sup, 
                         intervallo_inf, 
                         color=COLOR_ORIG, alpha=.25)
        plt.legend(loc='best');
        plt.show()
        
        # Calcoliamo le metriche d'errore identificando l'errore
        errore = ts_forecast - serie_totale
        errore.dropna(inplace=True)
        
        sommaPrevOss = ts_forecast + serie_totale
        sommaPrevOss.dropna(inplace=True)
        
        print(">>>>>> Calcoliamo MAE=%.4f"%(sum(abs(errore))/len(errore)))
        print(">>>>>> Calcoliamo MSE=%.4f"%(sum(errore**2)/len(sommaPrevOss)))