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
def autocorrelation(ts, lags):
    """
    Parameters
    ----------
    ts : pd.Series
        Serie temporale
    lags : integer
        Ampiezza finestra di visualizzazione del grafico di autocorrelazione
    Returns
    -------
    None.
    """
    autocor = acf(ts, nlags=lags)
    plt.plot(autocor, color = 'orange')
    #Delimito i tre intervalli
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='black')
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='black')
    plt.title('Funzione di autocorrelazione')
    
# %%
if __name__ == '__main__':
    print('Caricamento sales_train_validation.csv ...', end=' ')
    sales_train = load_data('./datasets/sales_train_validation.csv')
    print('Carimento completato')
    
    #%%
    
    shopNames = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    stateNames = ['CA', 'TX', 'WI']
    catNames = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
    
    # %%
    
    print('Creazione serie temporali (ancora dataframe) ...', end=' ')
    
    hobby = sales_train[sales_train['cat_id'] == 'HOBBIES']
    household = sales_train[sales_train['cat_id'] == 'HOUSEHOLD']
    food = sales_train[sales_train['cat_id'] == 'FOODS']
    
    stateCA = sales_train[sales_train['state_id'] == 'CA']
    stateTX = sales_train[sales_train['state_id'] == 'TX']
    stateWI = sales_train[sales_train['state_id'] == 'WI']
    
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
    
    shopList = [shopCA1, shopCA2, shopCA3, shopCA4, shopTX1, shopTX2, shopTX3, shopWI1, shopWI2, shopWI3]
    stateList = [stateCA, stateTX, stateWI]
    catList = [hobby, household, food]
    
    print('Creazione completata')

    # %%
    # Definisco l'array delle colonne d_1, ...., d_1913
    giorni = []
    for column in stateCA:
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

    # %%
    # Serie temporali per stato
    
    tsVenditeStato = []
    
    print('Sto creando le serie temporali delle vendite per stato...', end=' ')
    for state in stateList:
        tsVenditeStato.append(pd.Series(data=sumrows(state, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')
    #%%
    
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
    
    # %%
    # Serie temporali per categoria
    
    tsVenditeCat = []
    
    print('Sto creando le serie temporali delle vendite per categoria...', end=' ')
    for cat in catList:
        tsVenditeCat.append(pd.Series(data=sumrows(cat, giorni), 
                                          index=pd.date_range(start=pd.Timestamp('2011-01-29'), periods=1913, freq='D')))
    print('Operazione completata')
    #%%
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
    