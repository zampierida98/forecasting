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
 
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_data(filename, indexData=False):
    if indexData:        
        dateparser = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d')
        dataframe = pd.read_csv(filename, index_col = 0, date_parser=dateparser)
    else:
        dataframe = pd.read_csv(filename)
    return dataframe

if __name__ == '__main__':
    print('Caricamento sales_train_validation.csv ...', end=' ')
    sales_train = load_data('./datasets/sales_train_validation.csv')
    print('Carimento completato')
    
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
    
    print('Creazione completata')
    
    