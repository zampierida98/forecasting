# -*- coding: utf-8 -*-
"""
Analisi dei dati di un negozio di abbigliamento
"""
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import statsmodels.api as sm
import datetime as dt
import warnings
import itertools


def load_data(filename):
    """
    Carica i dati da un file csv e rende l'indice di tipo datetime

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
    data = pd.read_csv(filename, index_col=0, date_parser=dateparse)
    return data


def ts_plot(timeseries):
    """
    Plot di una serie temporale

    Parameters
    ----------
    timeseries : Series
        la serie temporale

    Returns
    -------
    None.

    """
    plt.figure(figsize=(40, 20))
    plt.title(timeseries.name)
    plt.plot(timeseries)
    plt.show()


def test_stationarity(timeseries):
    """
    Dickey-Fuller Test

    Parameters
    ----------
    timeseries : Series
        la serie temporale

    Returns
    -------
    None.

    """
    print('Results of Dickey-Fuller Test for', timeseries.name, '\n')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def acf_pacf(timeseries):
    """
    Plot di autocorrelazione e autocorrelazione parziale

    Parameters
    ----------
    timeseries : Series
        la serie temporale

    Returns
    -------
    None.

    """
    lag_acf = acf(timeseries, nlags=20)
    lag_pacf = pacf(timeseries, nlags=20, method='ols')
    plt.figure(figsize=(40, 20))
    # plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function '+timeseries.name)
    # plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function '+timeseries.name)
    plt.tight_layout()
    plt.show()


def order_selection(timeseries):
    """
    Selezione degli ordini (p,d,q) e (P,D,Q,S) con pmdarima

    Parameters
    ----------
    timeseries : Series
        la serie temporale

    Returns
    -------
    list
        [(p,d,q), (P,D,Q,S)]

    """
    model = pm.auto_arima(timeseries, seasonal=True, m=12, suppress_warnings=True, trace=True)
    return [model.order, model.seasonal_order]


def iterative_order_selection(timeseries, min_order=2, max_order=5):
    """
    Selezione iterativa degli ordini (p,d,q) per ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    min_order : int
        l'ordine minimo da considerare (default 2)
    max_order : int
        l'ordine massimo da considerare (default 5)

    Returns
    -------
    tuple
        (p,d,q)

    """
    p = d = q = range(min_order, max_order+1)
    pdq = list(itertools.product(p, d, q))
    
    orders = []
    aics = []
    
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    for param in pdq:
        try:
            mod = ARIMA(timeseries, order=param)
            results = mod.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            orders.append(param)
            aics.append(results.aic)
        except:
            continue # ignore the parameter combinations that cause issues
    
    #min_aic = np.min(aics)
    index_min_aic = np.argmin(aics)
    return orders[index_min_aic]


def sarimax_forecasting(timeseries, h):
    """
    Forecasting con SARIMAX

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    h : int
        l'orizzonte

    Returns
    -------
    None.

    """
    [o, so] = order_selection(timeseries)
    mod = sm.tsa.statespace.SARIMAX(timeseries,
                                    order=o,
                                    seasonal_order=so,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    
    results = mod.fit()
    print(results.summary())
    
    results.plot_diagnostics(figsize=(40,20))
    plt.show()
    
    pred_uc = results.get_forecast(steps=h)
    pred_ci = pred_uc.conf_int()
    
    ax = timeseries.plot(label='Observed', figsize=(40,20))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    plt.legend()
    plt.show()


def arima_forecasting(timeseries, h):
    """
    Forecasting con ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    h : int
        l'orizzonte

    Returns
    -------
    DataFrames
        la serie temporale basata sui risultati del forecast sul modello ARIMA

    """
    o = iterative_order_selection(timeseries)
    mod = ARIMA(timeseries, order=o)
    results = mod.fit()
    print(results.summary())
    
    pred = results.forecast(h) # tuple (forecast, stderr, conf_int)
    
    future_dates = pd.date_range(start=timeseries.index[len(timeseries)-1], periods=h, freq='D')
    future_series = pd.Series(data=pred[0], index=future_dates)
    future_series_ci_min = pd.Series(data=pred[2][:, 0], index=future_dates)
    future_series_ci_max = pd.Series(data=pred[2][:, 1], index=future_dates)
    forecasts = pd.concat([future_series, future_series_ci_min, future_series_ci_max], axis=1)
    
    ax = timeseries.plot(label='Observed', figsize=(40,20))
    forecasts[0].plot(ax=ax, label='Forecast')
    ax.fill_between(forecasts.index,
                    forecasts[1],
                    forecasts[2], color='k', alpha=.25)
    plt.legend()
    plt.show()
    
    return forecasts


# %% Main
if __name__ == '__main__':
    data = load_data('Dati_Albignasego/Whole period.csv')
    # colonne: MAGLIE, CAMICIE, GONNE, PANTALONI, VESTITI, GIACCHE
    for col in data.columns:
        ts = data[col]
        ts_plot(ts)
        
        # stagionalità:
        decomposition = seasonal_decompose(ts, period=12)
        decomposition.seasonal.plot(figsize=(40,20), title='Stagionalità '+col, fontsize=14)
        
        # stazionarietà:
        test_stationarity(ts) # the test statistic is smaller than the 1% critical values so we can say with 99% confidence that ts is stationary
        
        # plot di autocorrelazione e autocorrelazione parziale:
        acf_pacf(ts)
        
        
    # forecasting con SARIMAX (basato sugli orders ottenuti da pmdarima):
    ts = data['MAGLIE']
    sarimax_forecasting(ts, 50)
    
    # forecasting con ARIMA (basato sugli orders ottenuti minimizzando AIC):
    arima_forecasting(ts, 50)
    
    # TODO: funzione accuracy
    train = data[:'2016']
    test = data['2017':]
    arima_forecasts = arima_forecasting(train['MAGLIE'], len(ts)-len(train))
    
    ax = ts.plot(label='Test', figsize=(40,20))
    arima_forecasts[0].plot(ax=ax, label='Train forecast', alpha=.7)
    ax.fill_between(arima_forecasts.index,
                    arima_forecasts[1],
                    arima_forecasts[2], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    mse = ((arima_forecasts[0] - test['MAGLIE']) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
