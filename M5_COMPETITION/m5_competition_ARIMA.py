# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:30:52 2020

@author: seba3
"""

import datetime
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# SOPPRIMIAMO I WARNING DI NUMPY
np.warnings.filterwarnings('ignore')

# IMPOSTAZIONI SUI GRAFICI
SMALL_SIZE = 32
MEDIUM_SIZE = 34
BIGGER_SIZE = 40
COLOR_ORIG = 'black'
COLOR_MODEL = 'green'
COLOR_FOREC = 'red'
COLOR_ACF = 'blue'
COLORPALETTE = ['red', 'gold', 'blue', 'green', 'purple', 'orange', 'black', 'lime', 'cyan', 'peru', 'gray']
 
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#fornire timeserie e rispettiva etichetta in ordine!

def plot(timeseries = [], labels = [], titolo=''):
    """
    Parameters
    ----------
    timeseries : TYPE, optional
        DESCRIPTION. The default is [].
    labels : TYPE, optional
        DESCRIPTION. The default is [].
    titolo : TYPE, optional
        DESCRIPTION. The default is ''.
    Returns
    -------
    None.
    """
    plt.figure(figsize=(80, 40), dpi=60)
    plt.title(str(titolo))
    plt.ylabel('Vendite')
    plt.xlabel('Data')
    i=0
    for serie in timeseries:
        plt.plot(serie, label = str(labels[i]), color = COLORPALETTE[i])
        i += 1
    plt.legend(loc='best')
    plt.show()
    return 

def plot_results(timeseries = [], labels = [], titolo=''):
    """
    TSC (training set color) : 
        'black'  
    VSC (validation set color) : 
        'black'  
    FC (forecast color) : 
        'red'    
    MRC (model results color) : 
        'green'  
    OLC (other lines color) : 
        'orange'

    Parameters
    ----------
    timeseries : timeseries[], optional
        DESCRIPTION. The default is []. Order: TSC, VSC, FC, MRC, OLC
    labels : str[], optional
        DESCRIPTION. The default is []. Order: TSC, VSC, FC, MRC, OLC
    titolo : TYPE, optional
        DESCRIPTION. The default is ''.
    Returns
    -------
    None.
    """
    RESCOLORPALETTE = ['black','black','red','green','orange']
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(str(titolo))
    plt.ylabel('Vendite')
    plt.xlabel('Data')
    i=0
    for serie in timeseries:
        if i==1: # validation set
            plt.plot(serie, label = str(labels[i]), color = RESCOLORPALETTE[i], linestyle = '--')
        else:
            plt.plot(serie, label = str(labels[i]), color = RESCOLORPALETTE[i])
        i += 1
    plt.legend(loc='best')
    plt.show()
    return 

def sumrows(dataframe, giorni):
    res = [0]
    for g in giorni:
        ind = len(res) - 1
        for value in dataframe[g]:
            res[ind] += value
        res.append(0)

    # Rimuoviamo l'ultimo elemento che è 0
    return res[:-1]

def load_data(filename, indexData=False):
    if indexData:        
        dateparser = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d')
        dataframe = pd.read_csv(filename, index_col = 0, date_parser=dateparser)
    else:
        dataframe = pd.read_csv(filename)
    return dataframe

def rolling(ts, w, meanOrStd=True):
    '''
    Parameters
    ----------
    ts : pd.Series
        Serie temporale
    w : integer
        Finestra della rolling
    meanOrStd : bool, optional
        True se rolling mean, False std. The default is True.
    Returns
    -------
    Rolling mean

    '''
    if meanOrStd:
        return ts.rolling(window=w).mean()
    return ts.rolling(window=w).std()

def autocorrelation(ts = [], lags = 20, titleSpec = ''):
    """
    Parameters
    ----------
    ts : pd.Series
        Lista di serie temporali
    lags : integer
        Ampiezza finestra di visualizzazione del grafico di autocorrelazione
    titleSpec : str
        Specifica le serie di cui si calcola l'autocorrelazione (di fatto una 
        parte del titolo del grafico...)
    Returns
    -------
    None.
    """
    autocor = []
    for timeserie in ts:
        autocor.append(acf(timeserie, nlags=lags))
    i = 0
    plt.figure(figsize=(80, 40), dpi=60)
    for fun in autocor:
        plt.plot(fun, color = COLORPALETTE[i])
        i += 1
    #Delimito i tre intervalli
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='black')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='black')
    plt.legend(loc = 'best')
    title = 'Funzione di autocorrelazione: ' + str(titleSpec) 
    plt.title(title)
    plt.show()
    
def pautocorrelation(ts = [], lags = 20, titleSpec = ''):
    """
    Parameters
    ----------
    ts : pd.Series
        Lista di serie temporali
    lags : integer
        Ampiezza finestra di visualizzazione del grafico di autocorrelazione
    titleSpec : str
        Specifica le serie di cui si calcola l'autocorrelazione (di fatto una 
        parte del titolo del grafico...)
    Returns
    -------
    None.
    """
    pautocor = []
    for timeserie in ts:
        pautocor.append(pacf(timeserie, nlags=lags))
    i = 0
    plt.figure(figsize=(80, 40), dpi=60)
    for fun in pautocor:
        plt.plot(fun, color = COLORPALETTE[i])
        i += 1
    #Delimito i tre intervalli
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='black')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='black')
    plt.legend(loc = 'best')
    title = 'Funzione di autocorrelazione parziale: ' + str(titleSpec) 
    plt.title(title)
    plt.show()

def correlation(ts1, ts2):
    '''
    Correlation è il metodo che calcola la correlazione fra due serie temporali
    Parameters
    ----------
    ts1 : pd.Series
        Prima serie temporale
    ts2 : pd.Series
        Seconda serie temporale

    Returns res
    -------
    float
        È un valore compreso fra 0 e 1

    '''
    return ts1.corr(ts2)
"""
def ETS_FORECASTING(ts,periodo=365, h=100):
    '''
    La funzione ETS_FORECASTING calcola, le previsioni serie temporali
    ritornando il modello e le previsioni
    Parameters
    ----------
    ts : pd.Series
        Serie temporale
    periodo : int, optional
        Il periodo di stagionalità. The default is 365.
    h : int, optional
        Orizzonte di previsione
    Returns: 
    -------
    (model, model_forecasting) ovvero il modello e le previsioni
    '''
    model = ExponentialSmoothing(ts, trend="add", damped=True, seasonal="add", seasonal_periods=periodo)
    model_fitted = model.fit()
    model_forecasting = model_fitted.forecast(steps=h)
    return model_fitted.fittedvalues, model_forecasting
"""

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
        if PACF[i] <= limite or i >= 6:
            p = i
            break
    for i in range(0, len(ACF)):
        if ACF[i] <= limite or i >= 4:
            q = i
            break
    return (p,q)

def ARIMA_DECOMPOSITION_FORECASTING(ts, periodo=365, h=100):
    '''
    La funzione ARIMA_FORECASTING calcola le previsioni delle serie temporali
    ritornando il modello e le previsioni
    Parameters
    ----------
    ts : pd.Series
        Serie temporale
    periodo : int, optional
        Il periodo di stagionalità. The default is 365.
    h : int, optional
        Orizzonte di previsione. The default is 100.
    Returns: 
    -------
    (model, model_forecasting) ovvero il modello e le previsioni
    '''
    decomposition = seasonal_decompose(ts, period=periodo)
    
    # salvo le parti decomposte in variabili 
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # rimuovo i valori null
    trend.dropna(inplace=True)
    seasonal.dropna(inplace=True)
    residual.dropna(inplace=True)    

    # Uso ARIMA su ogni componente della decomposizione. Cerco gli ordini 
    # di p e q ed applico ARIMA
    p, q = p_q_for_ARIMA(trend)
    flag = True
    while flag:
        try:
            try:
                trend_model = ARIMA(trend, order = (p, 0, q))
            except:
                trend_model = ARIMA(trend, order = (p, 1, q))
            trend_fitted = trend_model.fit(transparams=False)
            flag = False
        except:
            q -= 1
    
    p, q = p_q_for_ARIMA(seasonal)
    flag = True
    while flag:
        try:
            try:
                seasonal_model = ARIMA(seasonal, order = (p, 0, q))
            except:
                seasonal_model = ARIMA(seasonal, order = (p, 1, q))
            seasonal_fitted = seasonal_model.fit(transparams=False)
            flag = False
        except:
            q -= 1
        
    flag = True
    q = 5
    residual_fitted = None
    while flag:
        try:    
            residual_model = ARIMA(residual, order=(5, 0, q))
            #fit model
            residual_fitted = residual_model.fit()
            flag = False
        except:
            q -= 1
            
    # make prediction. Stesso periodo del validation set!    
    trend_model_predictions_array, _, _ = trend_fitted.forecast(steps=h)
    seasonal_model_predictions_array, _, _ = seasonal_fitted.forecast(steps=h)
    residual_model_predictions_array, _, _ = residual_fitted.forecast(steps=h)
    
    trend_model_predictions = pd.Series(data=seasonal_model_predictions_array,
                                           index=pd.date_range(start=pd.Timestamp('2016-04-25'), periods=28, freq='D'))
    seasonal_model_predictions = pd.Series(data=trend_model_predictions_array, 
                                           index=pd.date_range(start=pd.Timestamp('2016-04-25'), periods=28, freq='D'))
    residual_model_predictions = pd.Series(data=residual_model_predictions_array, 
                                           index=pd.date_range(start=pd.Timestamp('2016-04-25'), periods=28, freq='D'))
    
    #Sommo i modelli
    model = trend_fitted.fittedvalues \
                + seasonal_fitted.fittedvalues \
                + residual_fitted.fittedvalues

    #Sommo le previsioni
    model_forecasting = trend_model_predictions \
                        + seasonal_model_predictions \
                        + residual_model_predictions                       

    return (model, model_forecasting)

def find_best_model(ts, d=0, max_p=6, max_q=5):
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


def ARIMA_DECOMPOSITION_FORECASTING_2(ts, periodo=365, h=100):
    '''
    La funzione ARIMA_FORECASTING calcola le previsioni delle serie temporali
    ritornando il modello e le previsioni
    Parameters
    ----------
    ts : pd.Series
        Serie temporale
    periodo : int, optional
        Il periodo di stagionalità. The default is 365.
    h : int, optional
        Orizzonte di previsione. The default is 100.
    Returns: 
    -------
    (model, model_forecasting) ovvero il modello e le previsioni
    '''
    decomposition = seasonal_decompose(ts, period=periodo)
    
    # salvo le parti decomposte in variabili 
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # rimuovo i valori null
    trend.dropna(inplace=True)
    seasonal.dropna(inplace=True)
    residual.dropna(inplace=True)    

    _,_,trend_fitted = find_best_model(trend, d=1)
    _,_,seasonal_fitted = find_best_model(seasonal, d=0)
        
    flag = True
    q = 5
    residual_fitted = None
    while flag:
        try:    
            residual_model = ARIMA(residual, order=(5, 0, q))
            #fit model
            residual_fitted = residual_model.fit()
            flag = False
        except:
            q -= 1
            
    # make prediction. Stesso periodo del validation set!    
    trend_model_predictions_array, _, _ = trend_fitted.forecast(steps=h)
    seasonal_model_predictions_array, _, _ = seasonal_fitted.forecast(steps=h)
    residual_model_predictions_array, _, _ = residual_fitted.forecast(steps=h)
    
    trend_model_predictions = pd.Series(data=seasonal_model_predictions_array,
                                           index=pd.date_range(start=pd.Timestamp('2016-04-25'), periods=28, freq='D'))
    seasonal_model_predictions = pd.Series(data=trend_model_predictions_array, 
                                           index=pd.date_range(start=pd.Timestamp('2016-04-25'), periods=28, freq='D'))
    residual_model_predictions = pd.Series(data=residual_model_predictions_array, 
                                           index=pd.date_range(start=pd.Timestamp('2016-04-25'), periods=28, freq='D'))
    
    #Sommo i modelli
    model = trend_fitted.fittedvalues \
                + seasonal_fitted.fittedvalues \
                + residual_fitted.fittedvalues

    #Sommo le previsioni
    model_forecasting = trend_model_predictions \
                        + seasonal_model_predictions \
                        + residual_model_predictions                       

    return (model, model_forecasting)

def MAE_error(ts, model):
    errore = model - ts
    errore.dropna(inplace=True)

    return sum(abs(errore))/len(errore)

def HyndmanAndKoehler_error(ts, model, periodo=365):
    errore = model - ts
    errore.dropna(inplace=True)
    
    T = len(ts)
    denominatore = 0
    for i in range(periodo, T):
        denominatore += abs(ts[i] - ts[i - periodo])
    denominatore *= 1/(T - periodo)
    
    q = []
    for ej in errore:
        q.append(abs(ej)/denominatore)
    
    res = 0
    for i in range(0,len(q)):
        res += q[i]
    return res/len(q)

def save_obj(obj, filename):
    '''
    save_list salva una lista su un file in filename 
    Parameters
    ----------
    l : list
        lista da salvare
    filename : str
        path del file

    Returns
    -------
    None.
    '''
    import pickle

    with open(filename, 'wb') as fout:
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    '''
    load_list recupera su lista i valori in un file
    Parameters
    ----------
    filename : str
        path del file da caricare

    Returns
    -------
    None.

    '''
    import pickle
    res = None
    with open(filename, 'rb') as fin:
        res = pickle.load(fin)
    return res
    
# %%
if __name__ == '__main__':
    print('Caricamento sales_train_validation.csv ...', end=' ')
    sales_train = load_data('./datasets/sales_train_validation.csv')
    print('Carimento completato')

    # Sezione dove definiamo i nomi delle proprietà/campi del file sales_train_validation.csv
    shopNames = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    stateNames = ['CA', 'TX', 'WI']
    catNames = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
    StatoAndCatNames = ['CA_HOBBIES', 'CA_HOUSEHOLD', 'CA_FOODS', 'TX_HOBBIES', 'TX_HOUSEHOLD', 'TX_FOOD',
                        'WI_HOBBIES', 'WI_HOUSEHOLD', 'WI_FOODS']

    # Sezione dove andiamo a creare i DATAFRAME che rappresenteranno le serie temporali nelle sezioni successive
    print('Creazione serie temporali (ancora dataframe) ...', end=' ')
    
    # DATAFRAME per categoria
    hobby = sales_train[sales_train['cat_id'] == 'HOBBIES']
    household = sales_train[sales_train['cat_id'] == 'HOUSEHOLD']
    food = sales_train[sales_train['cat_id'] == 'FOODS']

    # DATAFRAME per negozio
    shopCA1 = sales_train[sales_train['store_id'] == 'CA_1']
    shopCA2 = sales_train[sales_train['store_id'] == 'CA_2']
    shopCA3 = sales_train[sales_train['store_id'] == 'CA_3']
    shopCA4 = sales_train[sales_train['store_id'] == 'CA_4']
    
    shopTX1 = sales_train[sales_train['store_id'] == 'TX_1']
    shopTX2 = sales_train[sales_train['store_id'] == 'TX_2']
    shopTX3 = sales_train[sales_train['store_id'] == 'TX_3']
    
    shopWI1 = sales_train[sales_train['store_id'] == 'WI_1']
    shopWI2 = sales_train[sales_train['store_id'] == 'WI_2']
    shopWI3 = sales_train[sales_train['store_id'] == 'WI_3']
    
    # DATAFRAME per stato
    stateCA = sales_train[sales_train['state_id'] == 'CA']
    stateTX = sales_train[sales_train['state_id'] == 'TX']
    stateWI = sales_train[sales_train['state_id'] == 'WI']

    # DATAFRAME per stato e categoria
    stateCAhobbies = sales_train[np.logical_and(sales_train['state_id'] == 'CA', sales_train['cat_id'] == 'HOBBIES')]
    stateCAhousehold = sales_train[np.logical_and(sales_train['state_id'] == 'CA', sales_train['cat_id'] == 'HOUSEHOLD')]
    stateCAfoods = sales_train[np.logical_and(sales_train['state_id'] == 'CA', sales_train['cat_id'] == 'FOODS')]
    
    stateTXhobbies = sales_train[np.logical_and(sales_train['state_id'] == 'TX', sales_train['cat_id'] == 'HOBBIES')]
    stateTXhousehold = sales_train[np.logical_and(sales_train['state_id'] == 'TX', sales_train['cat_id'] == 'HOUSEHOLD')]
    stateTXfoods = sales_train[np.logical_and(sales_train['state_id'] == 'TX', sales_train['cat_id'] == 'FOODS')]
    
    stateWIhobbies = sales_train[np.logical_and(sales_train['state_id'] == 'WI', sales_train['cat_id'] == 'HOBBIES')]
    stateWIhousehold = sales_train[np.logical_and(sales_train['state_id'] == 'WI', sales_train['cat_id'] == 'HOUSEHOLD')]
    stateWIfoods = sales_train[np.logical_and(sales_train['state_id'] == 'WI', sales_train['cat_id'] == 'FOODS')]    
    
    #DATAFRAME per negozio e categoria
    shopAndCatList = []
    for s in shopNames:
        storeShobbies = sales_train[np.logical_and(sales_train['store_id'] == s, sales_train['cat_id'] == 'HOBBIES')]
        storeShousehold = sales_train[np.logical_and(sales_train['store_id'] == s, sales_train['cat_id'] == 'HOUSEHOLD')]
        storeSfoods = sales_train[np.logical_and(sales_train['store_id'] == s, sales_train['cat_id'] == 'FOODS')]
        shopAndCatList.append(storeShobbies)
        shopAndCatList.append(storeShousehold)
        shopAndCatList.append(storeSfoods)
    
    # LISTE DI DATAFRAME RAGGRUPPATE PER PROPRIETA'
    shopList = [shopCA1, shopCA2, shopCA3, shopCA4, shopTX1, shopTX2, shopTX3, shopWI1, shopWI2, shopWI3]
    stateList = [stateCA, stateTX, stateWI]
    catList = [hobby, household, food]
    stateAndCatList = [stateCAhobbies, stateCAhousehold, stateCAfoods, stateTXhobbies, stateTXhousehold,
                       stateTXfoods, stateWIhobbies, stateWIhousehold, stateWIfoods]
    
    print('Creazione completata')
    
    # Definisco l'array delle colonne d_1, ...., d_1913
    
    giorni = []
    for column in stateCA: # PRENDIAMO UN DATAFRAME FRA QUELLE SOPRA CONDIVIDONO LE COLONNE d_1,..., d_1913
        if 'd_' in column:
            giorni.append(column)
    # %%
    # Serie temporali per negozio

    # Trasformiamo in serie temporali   
    tsVenditeNegozio = []
    
    print('Sto creando le serie temporali delle vendite per negozio...', end=' ')
    for shop in shopList:
        tsVenditeNegozio.append(pd.Series(data=sumrows(shop, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')

    # Analiziamo le serie temporali dei negozi
    # Plottiamo la rolling mean (visualizziamo la componente TREND)
    
    rollingVenditeNegozio = []
    
    print('Genero le rolling mean per negozio... ', end=' ')
    for i in range(len(tsVenditeNegozio)):
        rollingVenditeNegozio.append(rolling(tsVenditeNegozio[i], w=7))
    print('Operazione completata')
    
    """
    print('Plot del grafico...', end=' ')
    plot(rollingVenditeNegozio, shopNames, 'Rolling mean vendite per negozio con window=%d'%7)
    print('Operazione completata') 
    """
    
    # Calcolo l'autocorrelazioni delle serie di vendite per negozio
    autocorrelation(tsVenditeNegozio, titleSpec = "Vendite per negozio", lags = 60)

    # %%
    # Serie temporali per stato
    
    tsVenditeStato = []
    
    print('Sto creando le serie temporali delle vendite per stato...', end=' ')
    for state in stateList:
        tsVenditeStato.append(pd.Series(data=sumrows(state, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')

    # Analiziamo le serie temporali per stato
    # Plottiamo la rolling mean (visualizziamo la componente TREND)
    
    rollingVenditeStato = []
    
    print('Genero le rolling mean per stato... ', end=' ')
    for i in range(len(tsVenditeStato)):
        rollingVenditeStato.append(rolling(tsVenditeStato[i], w=7))
    print('Operazione completata')
    
    """
    print('Plot del grafico...', end=' ')
    plot(rollingVenditeStato, stateNames, 'Rolling mean vendite per stato con window=%d'%7)
    print('Operazione completata')
    """
    
    # Calcolo l'autocorrelazioni delle serie di vendite per stato
    autocorrelation(tsVenditeStato, titleSpec = "Vendite per stato", lags = 400)
    
    # %%
    # Serie temporali per categoria
    
    tsVenditeCat = []
    
    print('Sto creando le serie temporali delle vendite per categoria...', end=' ')
    for cat in catList:
        tsVenditeCat.append(pd.Series(data=sumrows(cat, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')

    # Analiziamo le serie temporali per categoria
    # Plottiamo la rolling mean (visualizziamo la componente TREND)
    
    rollingVenditeCat = []
    
    print('Genero le rolling mean per categoria... ', end=' ')
    for i in range(len(tsVenditeCat)):
        rollingVenditeCat.append(rolling(tsVenditeCat[i], w=7))
    print('Operazione completata')
    
    """
    print('Plot del grafico...', end=' ')
    plot(rollingVenditeCat, catNames, 'Rolling mean vendite per categoria con window=%d'%7)
    print('Operazione completata')
    """
    
    # Calcolo l'autocorrelazioni delle serie di vendite per categoria
    autocorrelation(tsVenditeCat, titleSpec = "Vendite per categoria", lags = 400)

    #%%
    # Serie temporali per stato & categoria
    
    tsVenditeStatoAndCat = []
    
    print('Sto creando le serie temporali delle vendite per stato e categoria...', end=' ')
    for stateAndCat in stateAndCatList:
        tsVenditeStatoAndCat.append(pd.Series(data=sumrows(stateAndCat, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')

    # Analiziamo le serie temporali per stato & categoria
    # Plottiamo la rolling mean (visualizziamo la componente TREND)
    
    rollingVenditeStatoAndCat = []
    
    print('Genero le rolling mean per stato e categoria... ', end=' ')
    for i in range(len(tsVenditeStatoAndCat)):
        rollingVenditeStatoAndCat.append(rolling(tsVenditeStatoAndCat[i], w=7))
    print('Operazione completata')
   
    """
    print('Plot del grafico...', end=' ')
    plot(rollingVenditeStatoAndCat, StatoAndCatNames, 'Rolling mean vendite per stato e categoria con window=%d'%7)
    print('Operazione completata')
    """
    
    # Calcolo l'autocorrelazioni delle serie di vendite per categoria
    autocorrelation(tsVenditeStatoAndCat, titleSpec = "Vendite per stato e categoria", lags = 30)
    
    #%%
    # Serie temporali per negozio & categoria
    
    tsVenditeNegozioAndCat = []
    
    print('Sto creando le serie temporali delle vendite per negozio e categoria...', end=' ')
    for shopAndCat in shopAndCatList:
        tsVenditeNegozioAndCat.append(pd.Series(data=sumrows(shopAndCat, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')

    # Analizziamo le serie temporali per negozio & categoria
    
    rollingVenditeNegozioAndCat = []
    
    print('Genero le rolling mean per negozio e categoria... ', end=' ')
    for i in range(len(tsVenditeNegozioAndCat)):
        rollingVenditeNegozioAndCat.append(rolling(tsVenditeNegozioAndCat[i], w=7))
    print('Operazione completata')
    
    # %%
    # Calcolo la correlazione fra le serie temporali vendite per stato e per categoria
    '''
    ind = 0
    for s in tsVenditeStato:
        ind2 = 0
        for s2 in tsVenditeCat:
            print(stateNames[ind], catNames[ind2] , ':', round(correlation(s,s2),3))
            ind2 += 1
        ind += 1
        
    # Calcolo la correlazione fra le serie temporali vendite per negozio e per categoria
    
    ind = 0
    for s in tsVenditeNegozio:
        ind2 = 0
        for s2 in tsVenditeCat:
            print(shopNames[ind], catNames[ind2] , ':', round(correlation(s,s2),3))
            ind2 += 1
        ind += 1
    
    # Calcolo la correlazione fra le serie temporali vendite per stato e per negozio
    
    ind = 0
    for s in tsVenditeStato:
        ind2 = 0
        for s2 in tsVenditeNegozio:
            print(stateNames[ind], shopNames[ind2] , ':', round(correlation(s,s2),3))
            ind2 += 1
        ind += 1
    '''
    
    # %%
    
    # NOTA: VEDERE FIGURA PG.4 del doc M5-Competitors-Guide-Final-10-March-2020.docx

    # NOTA2: la sezione successiva va a caricare l'oggetto tsForecastingNegozio
        # da un file così non c'è ogni volta da aspettare che si generino le previsioni
        # visto che è lungo
    


    # Abbiamo la seguente gerarchia:
    # +(liv. 0)      vendite TOTALI
    # +(liv. 1)      vendite per STATO
    # +(liv. 2)      vendite per NEGOZIO
    # -(liv. 3)      vendite per CATEGORIA          <<< --- DA ELIMINARE
    # *(liv. 4)      vendite per STATO & CATEGORIA  <<< --- raggruppando ci trova le vendite per STATO SALTANDO NEGOZIO
    # +(liv. 5)      vendite per NEGOZIO & CATEGORIA
    
    
    # Quindi noi usando l'approccio bottom-up prima realizziamo le previsioni
    # delle serie temporali per NEGOZIO & CATEGORIA poi raggruppiamo le serie temporali
    # per NEGOZIO ottenendo "vendite per NEGOZIO" poi raggruppando ulteriormente
    # abbiamo quelle per "STATO" e infine le vendite "TOTALI"
    
    # partiamo da NEGOZIO & CATEGORIA
    """
    print('Stime modelli delle previsioni per NEGOZIO & CATEGORIA con ARIMA...')
    
    ind = 0
    j = 0
    tsForecastingNegozioAndCat = []
    for ts in tsVenditeNegozioAndCat:
        model,forecasting = ARIMA_DECOMPOSITION_FORECASTING_2(ts, periodo=7, h=1941-1913)
        mase = HyndmanAndKoehler_error(ts, model)
        print(f'MASE ARIMA_DECOMPOSITION_FORECASTING DI {shopNames[j]}_{catNames[ind%3]} = {mase}')
        tsForecastingNegozioAndCat.append(forecasting)
        if (ind+1)%3 == 0:
            j += 1
        ind += 1
        
    print('Operazione completata')
    
    # %%
    print('Salvo l\'oggetto "tsForecastingNegozioAndCatARIMA" su file così da caricarlo in momenti successivi')
    
    save_obj(tsForecastingNegozioAndCat, 'tsForecastingNegozioAndCatARIMA.pyobj')
    """
    # %%
    tsForecastingNegozioAndCat = load_obj('tsForecastingNegozioAndCatARIMA.pyobj')
    
    print('Caricamento di "tsForecastingNegozioAndCatARIMA" completato')
    
    # %%
    """
    print('Stime modelli delle previsioni per NEGOZIO...')
    
    ind = 0
    tsForecastingNegozio = []
    for ts in tsVenditeNegozio:
        model,forecasting = ETS_DECOMPOSITION_FORECASTING(ts,periodo=365, h=1941-1913)
        mase = HyndmanAndKoehler_error(ts, model)
        print(f'MASE ETS_DECOMPOSITION_FORECASTING DI {shopNames[ind]} = {mase}')
        tsForecastingNegozio.append(forecasting)
        ind+=1
        
    print('Operazione completata')
    
    print('Salvo l\'oggetto "tsForecastingNegozio" su file così da caricarlo in momenti successivi')
    save_obj(tsForecastingNegozio, 'tsForecastingNegozio.pyobj')

    tsForecastingNegozio = load_obj('tsForecastingNegozio.pyobj')
    print('Caricamento di "tsForecastingNegozio" completato')
    """
    # %%
    
    print("Raggruppiamo le serie temporali per NEGOZIO...")
    
    tsForecastingNegozio = []
    i = 0
    j = 0
    while i < len(tsForecastingNegozioAndCat):
        tsForecastingNegozio.append(tsForecastingNegozioAndCat[i])
        tsForecastingNegozio[j] += tsForecastingNegozioAndCat[i+1]
        tsForecastingNegozio[j] += tsForecastingNegozioAndCat[i+2]
        i = i+3
        j += 1
    
    print('Operazione completata')
    
    # %%
    
    print('Sommiamo le previsioni in base all\'appartenenza di un negozio ad uno STATO...')
    
    ts_Ger_ForecastingStato = []
    state = shopNames[0] + 'ABCD'     # stringa non presente nei nomi dei negozi
    ind = 0
    
    for s in shopNames:
        if state in s:
            ts_Ger_ForecastingStato[len(ts_Ger_ForecastingStato) - 1] += tsForecastingNegozio[ind]
        else:
            state = s[0:2]
            ts_Ger_ForecastingStato.append(tsForecastingNegozio[ind])
            
        ind += 1
    
    print('Operazione completata')
    
    print('Sommiamo le previsioni per ogni STATO ottenendo le previsioni di vendita TOTALI...')
    
    ts_Ger_ForecastingVenditeTot = ts_Ger_ForecastingStato[0][:]    #[:] per eseguire una copia
    ts_Ger_ForecastingVenditeTot += ts_Ger_ForecastingStato[1]
    ts_Ger_ForecastingVenditeTot += ts_Ger_ForecastingStato[2]
    
    print('Operazione completata')
    
    # %%
    
    print('Grafico delle previsioni per le vendite totali...')
    
    # operazioni per estrarre i dati reali da sales_train_evaluation.csv
    sales_train_evaluation = load_data('./datasets/sales_train_evaluation.csv')
    stateCAeval = sales_train_evaluation[sales_train_evaluation['state_id'] == 'CA']
    stateTXeval = sales_train_evaluation[sales_train_evaluation['state_id'] == 'TX']
    stateWIeval = sales_train_evaluation[sales_train_evaluation['state_id'] == 'WI']
    stateListEval = [stateCAeval, stateTXeval, stateWIeval]
    
    giorni = []
    for column in stateCAeval:
        if 'd_' in column:
            giorni.append(column)
    
    tsVenditeStatoEval = []
    for state in stateListEval:
        tsVenditeStatoEval.append(pd.Series(data=sumrows(state, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1941, freq='D')))
    
    tsVenditeTotEval = tsVenditeStatoEval[0][:]
    tsVenditeTotEval += tsVenditeStatoEval[1]
    tsVenditeTotEval += tsVenditeStatoEval[2]
    
    # ultimi 28 giorni
    tsVenditeTotValSet = tsVenditeTotEval[1913:]
    
    # creo il grafico
    tsVenditeTot = tsVenditeStato[0][:]
    tsVenditeTot += tsVenditeStato[1]
    tsVenditeTot += tsVenditeStato[2]
    plot_results([tsVenditeTot['2015-01-01':], tsVenditeTotValSet, ts_Ger_ForecastingVenditeTot], ['vendite totali', 'set di valutazione', 'previsioni'], 'Previsioni con ARIMA (miglior modello) per le vendite totali (bottom-up)')

    # metriche di errore
    errore = ts_Ger_ForecastingVenditeTot - tsVenditeTotValSet
    #errore.dropna(inplace=True)
    print('RMSE=%.4f'%np.sqrt((errore ** 2).mean()))
    print('MAE=%.4f'%(abs(errore)).mean())
    print('MAPE=%.4f'%(abs(100*errore/tsVenditeTotValSet)).mean())

    print('Operazione completata')
    
    # %%
    
    print('Forecast diretto sulle vendite totali...')
    
    model, tsForecastingVenditeTot = ARIMA_DECOMPOSITION_FORECASTING_2(tsVenditeTot, periodo=7, h=1941-1913)
    mase = HyndmanAndKoehler_error(tsVenditeTot, model)
    print(f'MASE ARIMA_DECOMPOSITION_FORECASTING_2 DI VENDITE TOTALI = {mase}')
    
    plot_results([tsVenditeTot['2015-01-01':], tsVenditeTotValSet, tsForecastingVenditeTot], ['vendite totali', 'set di valutazione', 'previsioni'], 'Previsioni con ARIMA (miglior modello) per le vendite totali (diretto)')
    
    # metriche di errore
    errore = tsForecastingVenditeTot - tsVenditeTotValSet
    #errore.dropna(inplace=True)
    print('RMSE=%.4f'%np.sqrt((errore ** 2).mean()))
    print('MAE=%.4f'%(abs(errore)).mean())
    print('MAPE=%.4f'%(abs(100*errore/tsVenditeTotValSet)).mean())
    
    # grafico comparativo
    plot_results([tsVenditeTot['2016-03-01':], tsVenditeTotValSet, ts_Ger_ForecastingVenditeTot, tsForecastingVenditeTot], ['vendite totali', 'set di valutazione', 'predizione con aggregazione', 'predizione sul totale'], 'Previsioni con ARIMA (miglior modello) per le vendite totali (comparazione)')
    
    print('Operazione completata')