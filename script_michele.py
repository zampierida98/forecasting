# -*- coding: utf-8 -*-
"""
Scipt di forecasting dati di 

5) git add
Utility : adds changes to stage/index in your working directory.
How to : git add .

6) git commit
Utility : commits your changes and sets it to new commit object for your remote.
How to : git commit -m”sweet little commit message”

7) git push/git pull
Utility : Push or Pull your changes to remote. If you have added and committed your changes and you want to push them. Or if your remote has updated and you want those latest changes.
How to : git pull <:remote:> <:branch:> and git push <:remote:> <:branch:>

"""
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA


def timeplot(ts, label='', linestyle='-'):
    '''
    Funzione che fa un plot di una serie temporale
    
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
    plt.figure(figsize=(12,6))
    plt.plot(ts)
    plt.plot(ts,  label=label, linestyle=linestyle)
    plt.legend(loc='best');
    plt.show()

def plotAcf(ts):
    plt.figure(figsize=(12,6))
    plt.plot(acf(ts))
    plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.axhline(y=0, linestyle="--", color="red")
    plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='red')
    plt.show()


if __name__ == "__main__":
    # COLONNE
    #,MAGLIE,CAMICIE,GONNE,PANTALONI,VESTITI,GIACCHE
    
    #Date già con YYYY-MM-DD
    #Whole period.csv
    dataframe = pd.read_csv('Dati_Albignasego/Whole period.csv',index_col = 0)
    
    # Analizziamo i dati graficamente
    
    '''
    for column in dataframe:
        timeplot(ts=dataframe[column], label=column)'''    
    # Commentato perchè impiega davvero tanto tempo
    
    # Lavoriamo intanto con le MAGLIE
    maglie = dataframe['MAGLIE']
    
    # Cerchiamo di capire se la serie è stagionale e/o presenta trend dalla funzione acf
    
    plotAcf(maglie)
    
    # Definiamo il training set 80% dati
    '''t_maglie = maglie[pd.date_range(start=dataframe.index[0], 
                                    end=maglie.index[int(len(maglie) * 0.8)] , freq='D')]'''
    