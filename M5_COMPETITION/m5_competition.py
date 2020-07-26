# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:51:56 2020

@author: michele
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
# %%
def plot(timeseries = [], labels = [], titolo=''):
    plt.figure(figsize=(80, 40), dpi=60)
    plt.title(str(titolo))
    plt.ylabel('Vendite')
    plt.xlabel('Data')
    i=0
    for serie in timeseries:
        plt.plot(serie, label = str(labels[i]), color = COLORPALETTE[i])
        i += 1
    plt.legend(loc='best')
    plt.show(block=False)
    return 
# %%
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

# %%
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

#%%
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

def ETS_DECOMPOSITION_FORECASTING(ts, periodo=365, due_lati=False, h=100):
    '''
    La funzione ETS_FORECASTING calcola, per decomposizione, le previsioni serie temporali
    ritornando il modello e le previsioni
    Parameters
    ----------
    ts : pd.Series
        Serie temporale
    periodo : int, optional
        Il periodo di stagionalità. The default is 365.
    due_lati : bool, optional
        Indica il two_sided della decomposizione. The default is False.
    
    h : int, optional
        Orizzonte di previsione
    Returns: 
    -------
    (model, model_forecasting) ovvero il modello e le previsioni
    '''
    decomposition = seasonal_decompose(ts, period=periodo, two_sided=due_lati)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    trend.dropna(inplace=True)
    seasonal.dropna(inplace=True)
    residual.dropna(inplace=True)    

    # Creiamo dei modelli per trend e seasonal + USO ARIMA PER I RESIDUAL VISTO CHE SONO UNA COMPONENTE STAZION.
    trend_model = ExponentialSmoothing(trend, trend="add", damped = True, seasonal=None)
    seasonal_model = ExponentialSmoothing(seasonal, trend=None, seasonal='add', seasonal_periods=periodo)

    # fit model
    trend_fitted    = trend_model.fit()
    seasonal_fitted = seasonal_model.fit()
    
    # ARIMA SU RESIDUAL (PER FORZA)
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
    trend_model_predictions = trend_fitted.forecast(steps=h)
    seasonal_model_predictions = seasonal_fitted.forecast(steps=h)
    residual_model_predictions, _, _ = residual_fitted.forecast(steps=h)
    
    #Sommo i modelli
    model = trend_fitted.fittedvalues \
                + seasonal_fitted.fittedvalues \
                + residual_fitted.fittedvalues

    #Sommo le previsioni
    model_forecasting = trend_model_predictions \
                        + seasonal_model_predictions \
                        + residual_model_predictions                       
    model_forecasting.dropna(inplace=True)
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
    
    #%% 
    # Sezione dove definiamo i nomi delle proprietà/campi del file sales_train_validation.csv
    shopNames = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    stateNames = ['CA', 'TX', 'WI']
    catNames = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
    StatoAndCatNames = ['CA_HOBBIES', 'CA_HOUSEHOLD', 'CA_FOODS', 'TX_HOBBIES', 'TX_HOUSEHOLD', 'TX_FOOD',
                        'WI_HOBBIES', 'WI_HOUSEHOLD', 'WI_FOODS']
    
    # %%
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
    # %%
    # Definisco l'array delle colonne d_1, ...., d_1913
    
    giorni = []
    for column in stateCA: # PRENDIAMO UN DATAFRAME FRA QUELLE SOPRA CONDIVIDONO LE COLONNE d_1,..., d_1913
        if 'd_' in column:
            giorni.append(column)
    # %%
    # Serie temporali per negozio

    # Trasformiamo in serie temporali
    # i negozi sono chiusi a Natale    
    tsVenditeNegozio = []
    
    print('Sto creando le serie temporali delle vendite per negozio...', end=' ')
    for shop in shopList:
        tsVenditeNegozio.append(pd.Series(data=sumrows(shop, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')
    #%%
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
    autocorrelation(tsVenditeNegozio, titleSpec = "Vendite per negozio", lags = 400)

    # %%
    # Serie temporali per stato
    
    tsVenditeStato = []
    
    print('Sto creando le serie temporali delle vendite per stato...', end=' ')
    for state in stateList:
        tsVenditeStato.append(pd.Series(data=sumrows(state, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')
    #%%
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
    #%%
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
    #%%
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
    #%%
    # Analizziamo le serie temporali per negozio & categoria
    
    rollingVenditeNegozioAndCat = []
    
    print('Genero le rolling mean per negozio e categoria... ', end=' ')
    for i in range(len(tsVenditeNegozioAndCat)):
        rollingVenditeNegozioAndCat.append(rolling(tsVenditeNegozioAndCat[i], w=7))
    print('Operazione completata')
    
    # %%
    # Calcolo la correlazione fra le serie temporali vendite per stato e per categoria
    
    ind = 0
    for s in tsVenditeStato:
        ind2 = 0
        for s2 in tsVenditeCat:
            print(stateNames[ind], catNames[ind2] , ':', round(correlation(s,s2),3))
            ind2 += 1
        ind += 1
    # %%
    # Calcolo la correlazione fra le serie temporali vendite per negozio e per categoria
    
    ind = 0
    for s in tsVenditeNegozio:
        ind2 = 0
        for s2 in tsVenditeCat:
            print(shopNames[ind], catNames[ind2] , ':', round(correlation(s,s2),3))
            ind2 += 1
        ind += 1
    
    # %%
    # Calcolo la correlazione fra le serie temporali vendite per stato e per negozio
    
    ind = 0
    for s in tsVenditeStato:
        ind2 = 0
        for s2 in tsVenditeNegozio:
            print(stateNames[ind], shopNames[ind2] , ':', round(correlation(s,s2),3))
            ind2 += 1
        ind += 1
    
    # %%
    # Usiamo ETS_FORECASTING e ETS_DECOMPOSITION_FORECASTING
    # PER PREVEDERE LE SERIE TEMPORALI
    # Serie per stato

    print('Stime modelli delle previsioni per STATO...')
    
    #plot([ts, model, forecasting], [stateNames[ind], 'modello', 'previsioni'], 'Previsioni con ETS per '+stateNames[ind])
    ind = 0
    for ts in tsVenditeStato:
        model,forecasting = ETS_DECOMPOSITION_FORECASTING(ts,periodo=365, h=1941-1913)
        mase = HyndmanAndKoehler_error(ts, model)
        print(f'MASE ETS_DECOMPOSITION_FORECASTING DI {stateNames[ind]} = {mase}')
        model,forecasting = ETS_FORECASTING(ts,periodo=365, h=1941-1913)
        mase = HyndmanAndKoehler_error(ts, model)
        print(f'MASE ETS_FORECASTING DI {stateNames[ind]} = {mase}')
        ind+=1
        
    print('Operazione completata')
    print('Meglio ETS con decomposizione che quello ETS(A,A)')
    
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
    
    # parto da NEGOZIO ma dovremmo partire da NEGOZIO & CATEGORIA
    
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
    
    # %%
    print('Salvo l\'oggetto "tsForecastingNegozio" su file così da caricarlo in momenti successivi')
    
    save_obj(tsForecastingNegozio, 'tsForecastingNegozio.pyobj')
    
    # %%
    tsForecastingNegozio = load_obj('tsForecastingNegozio.pyobj')
    
    print('Caricamento di "tsForecastingNegozio" completato')
    
    # %%
    
    print('Sommiamo le previsioni in base al\'appartenenza di un negozio ad uno STATO...')
    
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