# -*- coding: utf-8 -*-
"""
Analisi dei dati di un negozio di abbigliamento
"""
import datetime as dt
from fbprophet import Prophet
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tbats import TBATS
import warnings

SMALL_SIZE = 28
MEDIUM_SIZE = 30
BIGGER_SIZE = 32

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    plt.plot(timeseries, color='black')
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
    plt.figure(figsize=(40, 20), dpi=80)
    # plot ACF:
    plt.subplot(211)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.title('Autocorrelation Function '+timeseries.name)
    # plot PACF:
    plt.subplot(212)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.title('Partial Autocorrelation Function '+timeseries.name)
    plt.tight_layout()
    plt.show()


def order_selection(timeseries, m):
    """
    Selezione degli ordini (p,d,q) e (P,D,Q,S) con pmdarima

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    m : int
        numero di osservazioni in un anno

    Returns
    -------
    tuple
        ((p,d,q), (P,D,Q,S))

    """
    model = pm.auto_arima(timeseries, seasonal=True, m=m, suppress_warnings=True, trace=True)
    return (model.order, model.seasonal_order)


def sarimax_forecasting(timeseries, m, h):
    """
    Forecasting con SARIMAX

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    m : int
        numero di osservazioni in un anno
    h : int
        l'orizzonte

    Returns
    -------
    PredictionResultsWrapper
        le informazioni sui risultati del forecast sul modello SARIMAX

    """
    # realizzo il modello con gli ordini ottenuti da pmdarima
    (o, so) = order_selection(timeseries, m)
    mod = sm.tsa.statespace.SARIMAX(timeseries,
                                    order=o,
                                    seasonal_order=so,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    
    results = mod.fit()
    print(results.summary())
    
    # controllo la sua bontà
    results.plot_diagnostics(figsize=(40,20))
    plt.show()
    
    # predico h valori
    pred_uc = results.get_forecast(steps=h)
    pred_ci = pred_uc.conf_int()
    
    # grafico dei valori osservati e dei valori predetti
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    ax = timeseries.plot(label='Observed', color='black')
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='green')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    plt.legend()
    plt.show()
    
    return pred_uc


def iterative_order_selection(timeseries, min_order=0, max_order=4):
    """
    Selezione iterativa degli ordini (p,d,q) per ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    min_order : int
        l'ordine minimo da considerare (default 0)
    max_order : int
        l'ordine massimo da considerare (default 4)

    Returns
    -------
    tuple
        (p,d,q)

    """
    # calcolo tutte le possibili combinazioni di (p,d,q)
    p = d = q = range(min_order, max_order+1)
    pdq = list(itertools.product(p, d, q))
    
    # provo tutte le combinazioni
    orders = []
    aics = []
    warnings.filterwarnings("ignore")
    for param in pdq:
        try:
            mod = ARIMA(timeseries, order=param)
            results = mod.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            orders.append(param)
            aics.append(results.aic)
        except:
            continue
    
    # ritorno la combinazione che realizza il modello con AIC più basso
    index_min_aic = np.argmin(aics)
    return orders[index_min_aic]


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
    # realizzo il modello con gli ordini ottenuti dalla ricerca iterativa
    o = iterative_order_selection(timeseries)
    mod = ARIMA(timeseries, order=o)
    results = mod.fit()
    print(results.summary())
    
    # predico h valori
    pred = results.forecast(h) # ritorna una tupla (forecast, stderr, conf_int)
    
    # calcolo la serie temporale dei valori predetti
    future_dates = pd.date_range(start=timeseries.index[len(timeseries)-1], periods=h, freq='D')
    future_series = pd.Series(data=pred[0], index=future_dates)
    future_series_ci_min = pd.Series(data=pred[2][:, 0], index=future_dates)
    future_series_ci_max = pd.Series(data=pred[2][:, 1], index=future_dates)
    forecasts = pd.concat([future_series, future_series_ci_min, future_series_ci_max], axis=1)
    
    # grafico dei valori osservati e dei valori predetti
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    ax = timeseries.plot(label='Observed', color='black')
    forecasts[0].plot(ax=ax, label='Forecast', color='green')
    ax.fill_between(forecasts.index,
                    forecasts[1],
                    forecasts[2], color='k', alpha=.25)
    plt.legend()
    plt.show()
    
    return forecasts


def tbats_forecasting(timeseries, h, s):
    """
    Forecasting con TBATS

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    h : int
        l'orizzonte
    s : list
        l'array delle stagionalità

    Returns
    -------
    NumPy array
        i valori predetti

    """
    # fit the model
    estimator = TBATS(
        seasonal_periods=s,
        use_arma_errors=False,  # shall try only models without ARMA
        use_box_cox=False  # will not use Box-Cox
    )
    model = estimator.fit(timeseries)
    
    # fit the model (slow)
    #estimator_slow = TBATS(seasonal_periods=s)
    #model = estimator_slow.fit(timeseries)
    
    # summarize fitted model
    print(model.summary())
    
    return model.forecast(steps=h)


def fourier_forecasting(timeseries, m, end_train):
    """
    Forecasting con SARIMAX (with Fourier terms) e verifica dell'accuratezza

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    m : int
        numero di osservazioni in un anno
    end_train : str
        l'ultimo anno da considerare nel set di train

    Returns
    -------
    float
        MSE

    """
    # spezzo la serie temporale in due set (train e test)
    train = timeseries[:end_train]
    test = timeseries[str(int(end_train)+1):]
    
    # preparo le variabili esogene (Fourier terms)
    exog = pd.DataFrame({'date': timeseries.index})
    exog = exog.set_index(pd.PeriodIndex(exog['date'], freq='D'))
    exog['sin365'] = np.sin(2 * np.pi * exog.index.dayofyear / 365.25)
    exog['cos365'] = np.cos(2 * np.pi * exog.index.dayofyear / 365.25)
    exog['sin365_2'] = np.sin(4 * np.pi * exog.index.dayofyear / 365.25)
    exog['cos365_2'] = np.cos(4 * np.pi * exog.index.dayofyear / 365.25)
    exog = exog.drop(columns=['date'])
    
    exog_to_train = exog.iloc[:(len(timeseries)-len(test))]
    exog_to_test = exog.iloc[(len(timeseries)-len(test)):]
    
    # realizzo il modello con gli ordini ottenuti da pmdarima
    arima_exog_model = pm.auto_arima(y=train, exogenous=exog_to_train, seasonal=True, m=m, suppress_warnings=True, trace=True)
    
    # calcolo la serie temporale dei valori predetti
    arima_exog_forecast = arima_exog_model.predict(n_periods=len(test), exogenous=exog_to_test)
    future_dates = pd.date_range(start=ts.index[len(train)], periods=len(test), freq='D')
    future_series = pd.Series(data=arima_exog_forecast, index=future_dates)
    
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    ax = timeseries.plot(label='Observed', color='black')
    future_series.plot(ax=ax, label='Forecasted test', alpha=.7, color='green')
    #ax.fill_between(forecasts.index, forecasts[1], forecasts[2], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo del MSE
    mse = ((future_series[0] - timeseries) ** 2).mean()
    return round(mse, 2)


def prophet_forecasting(timeseries, h): # TODO: da sistemare
    """
    Forecasting con Prophet

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    h : int
        l'orizzonte

    Returns
    -------
    DataFrames
        la serie temporale basata sui risultati del forecast

    """
    df = pd.DataFrame(data=timeseries.to_numpy(), index=timeseries.index, columns=['y'])
    df.insert(0, 'ds', timeseries.index)
    
    m = Prophet()
    m.fit(df);
    future = m.make_future_dataframe(periods=h)
    forecast = m.predict(future)
    
    m.plot(forecast);
    m.plot_components(forecast);
    
    return forecast[len(timeseries):len(timeseries)+h]
    
    
def accuracy_sarimax(timeseries, m, train_length):
    """
    Verifica visuale dell'accuratezza di un modello SARIMAX

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    m : int
        numero di osservazioni in un anno
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)

    Returns
    -------
    float
        MSE

    """
    # spezzo la serie temporale
    train = timeseries[pd.date_range(start=data.index[0], end=timeseries.index[int(len(timeseries) * train_length)], freq='D')]

    # predico valori per la lunghezza del set di test
    forecasts = sarimax_forecasting(train, m, len(timeseries)-len(train))
    pred_ci = forecasts.conf_int()
    
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    ax = timeseries.plot(label='Observed', color='black')
    forecasts.predicted_mean.plot(ax=ax, label='Forecasted test', alpha=.7, color='green')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo del MSE
    mse = ((forecasts.predicted_mean[0] - timeseries) ** 2).mean()
    return round(mse, 2)


def accuracy_arima(timeseries, train_length):
    """
    Verifica visuale dell'accuratezza di un modello ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)

    Returns
    -------
    float
        MSE

    """
    # spezzo la serie temporale
    train = timeseries[pd.date_range(start=data.index[0], end=timeseries.index[int(len(timeseries) * train_length)], freq='D')]

    # predico valori per la lunghezza del set di test
    arima_forecasts = arima_forecasting(train, len(timeseries)-len(train))
    
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    ax = timeseries.plot(label='Observed', color='black')
    arima_forecasts[0].plot(ax=ax, label='Forecasted test', alpha=.7, color='green')
    ax.fill_between(arima_forecasts.index,
                    arima_forecasts[1],
                    arima_forecasts[2], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo del MSE
    mse = ((arima_forecasts[0] - timeseries) ** 2).mean()
    return round(mse, 2)


def accuracy_tbats(timeseries, forecasts):
    """
    Verifica visuale dell'accuratezza di un modello TBATS

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    forecasts : Series
        la serie dei valori predetti

    Returns
    -------
    float
        MSE

    """
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title(timeseries.name)
    ax = timeseries.plot(label='Observed', color='black')
    forecasts.plot(ax=ax, label='Forecasted test', alpha=.7, color='green')
    #ax.fill_between(forecasts.index, forecasts[1], forecasts[2], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo del MSE
    mse = ((forecasts[0] - timeseries) ** 2).mean()
    return round(mse, 2)
    

def accuracy_prophet(timeseries, end_train): # TODO: da sistemare
    """
    Verifica visuale dell'accuratezza di un modello ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    end_train : str
        l'ultimo anno da considerare nel set di train

    Returns
    -------
    float
        MSE

    """
    # spezzo la serie temporale in due set (train e test)
    train = timeseries[:end_train]
    test = timeseries[str(int(end_train)+1):]
    
    # predico valori per la lunghezza del set di test
    forecasts = prophet_forecasting(train, len(test))
    
    # calcolo la serie temporale dei valori predetti
    forecasts_dates = pd.date_range(start=ts.index[len(train)], periods=len(test), freq='D')
    yhat = pd.Series(data=forecasts['yhat'].to_numpy(), index=forecasts_dates)
    yhat_lower = pd.Series(data=forecasts['yhat_lower'].to_numpy(), index=forecasts_dates)
    yhat_upper = pd.Series(data=forecasts['yhat_upper'].to_numpy(), index=forecasts_dates)
    forecasts_series = pd.concat([yhat, yhat_lower, yhat_upper], axis=1)
    
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure()
    ax = timeseries.plot(label='Observed', figsize=(40,20))
    forecasts_series[0].plot(ax=ax, label='Forecasted test', alpha=.7)
    ax.fill_between(forecasts_series[0].index,
                    forecasts_series[1],
                    forecasts_series[2], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo del MSE
    mse = ((forecasts_series[0] - timeseries) ** 2).mean()
    return round(mse, 2)


if __name__ == '__main__':
    data = load_data('Dati_Albignasego/Whole period.csv')
    # colonne: MAGLIE, CAMICIE, GONNE, PANTALONI, VESTITI, GIACCHE
    ts = data['CAMICIE']
    
    # %% Analisi di stagionalità e stazionarietà
    for col in data.columns:
        ts = data[col]
        ts_plot(ts)
        
        # stagionalità:
        decomposition = seasonal_decompose(ts, period=12)
        decomposition.seasonal.plot(figsize=(40,20), title='Stagionalità '+col, fontsize=14)
        
        # plot di autocorrelazione e autocorrelazione parziale:
        acf_pacf(ts)
        
        # stazionarietà:
        test_stationarity(ts) # the test statistic is smaller than the 1% critical values so we can say with 99% confidence that ts is stationary
        
    # %% SARIMAX
    # forecasting con SARIMAX (basato sugli orders ottenuti da pmdarima):
    sarimax_forecasting(ts, 7, 50) # ignoro la stagionalità annuale
    
    # controllo l'accuratezza delle previsioni di SARIMAX confrontandole con la serie stessa:
    mse_sarimax = accuracy_sarimax(ts, 7, 0.8) # uso i dati fino alla fine del 2016 per prevedere i successivi
    
    # %% ARIMA
    # forecasting con ARIMA (basato sugli orders ottenuti minimizzando AIC):
    arima_forecasting(ts, 50)
    
    # controllo l'accuratezza delle previsioni di ARIMA confrontandole con la serie stessa:
    mse_arima = accuracy_arima(ts, 0.8) # uso i dati fino alla fine del 2016 per prevedere i successivi
    
    # %% TBATS
    ts_to_train = ts[pd.date_range(start=data.index[0], end=ts.index[int(len(ts) * 0.8)], freq='D')]
    
    # forecasting con TBATS:
    ts_forecast = tbats_forecasting(ts_to_train, len(ts)-len(ts_to_train), [7, 365.25])
    
    # calcolo la serie temporale dei valori predetti:
    future_dates = pd.date_range(start=ts.index[len(ts_to_train)], periods=len(ts)-len(ts_to_train), freq='D')
    future_series = pd.Series(data=ts_forecast, index=future_dates)
    
    # controllo l'accuratezza delle previsioni di TBATS confrontandole con la serie stessa:
    mse_tbats = accuracy_tbats(ts, future_series)
    
    # %% SARIMAX (with Fourier terms)
    mse_fourier = fourier_forecasting(ts, 7, 0.8)
    
    # %% Prophet
    # forecasting con Prophet:
    prophet_forecasting(ts, 50)
    
    # controllo l'accuratezza delle previsioni di Prophet confrontandole con la serie stessa:
    mse_prophet = accuracy_prophet(ts, '2016') # uso i dati fino alla fine del 2016 per prevedere i successivi
    
    # %% Comparazione dei modelli
    print("SARIMAX:", mse_sarimax)
    print("ARIMA:", mse_arima)
    print("TBATS:", mse_tbats)
    print("Fourier:", mse_fourier)
    