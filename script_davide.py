# -*- coding: utf-8 -*-
"""
Analisi dei dati di un negozio di abbigliamento
"""
import datetime as dt
from fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tbats import TBATS

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


# ==================================SARIMAX==================================
def sarimax_statsmodels(timeseries, train_length, m):
    """
    Realizzazione di un modello SARIMAX (con le funzioni di statsmodels)

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)
    m : int
        stagionalità
    
    Returns
    -------
    None.

    """
    # spezzo la serie temporale
    train = timeseries[pd.date_range(start=timeseries.index[0], end=timeseries.index[int(len(timeseries) * train_length)-1], freq='D')]

    # realizzo il modello
    model = pm.auto_arima(train, seasonal=True, m=m, suppress_warnings=True, trace=True,
                          start_p=1, start_q=1, max_p=2, max_q=2, start_P=1, start_Q=1, max_P=2, max_Q=2)
    print(model.summary())
    
    # controllo la sua bontà
    plt.figure(figsize=(40, 20), dpi=80)
    model.plot_diagnostics(figsize=(40, 20))
    plt.show()
    
    # ricavo il modello (predizioni in-sample)
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_prediction.html
    sarimax_mod = model.arima_res_.get_prediction(end=len(train)-1, dynamic=False)
    sarimax_dates = pd.date_range(start=timeseries.index[0], end=timeseries.index[len(train)-1], freq='D')
    sarimax_ts = pd.Series(sarimax_mod.predicted_mean, index=sarimax_dates)
    
    # grafico del modello ricavato
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Modello SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = train.plot(label='Train set', color='black')
    sarimax_ts.plot(ax=ax, label='In-sample predictions', color='green')
    plt.legend()
    plt.show()
    
    # out-of-sample forecasts
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast.html
    fcast = model.arima_res_.get_forecast(steps=len(timeseries)-len(train))
    fcast_ci = fcast.conf_int()
    
    fcast_dates = pd.date_range(start=timeseries.index[len(train)], periods=len(timeseries)-len(train), freq='D')
    ts_fcast = pd.Series(fcast.predicted_mean, index=fcast_dates)
    ts_ci_min = pd.Series(fcast_ci[:, 0], index=fcast_dates)
    ts_ci_max = pd.Series(fcast_ci[:, 1], index=fcast_dates)
    
    # grafico delle previsioni out-of-sample in sovrapposizione con i dati di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='Out-of-sample forecasts', alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo MAE e MSE
    errore = ts_fcast - timeseries
    errore.dropna(inplace=True)
    print('MSE=%.4f'%(errore ** 2).mean())
    print('MAE=%.4f'%(abs(errore)).mean())


def sarimax_pmdarima(timeseries, train_length, m):
    """
    Realizzazione di un modello SARIMAX (con le funzioni di pmdarima)

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)
    m : int
        stagionalità
    
    Returns
    -------
    None.

    """
    # spezzo la serie temporale
    train, test = model_selection.train_test_split(timeseries, train_size=train_length)

    # realizzo il modello
    model = pm.auto_arima(train, seasonal=True, m=m, suppress_warnings=True, trace=True,
                          start_p=1, start_q=1, max_p=2, max_q=2, start_P=1, start_Q=1, max_P=2, max_Q=2)
    print(model.summary())
    
    # ricavo il modello (predizioni in-sample)
    # http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.predict_in_sample
    preds = model.predict_in_sample(end=len(train)-1)
    sarimax_dates = pd.date_range(start=timeseries.index[0], end=timeseries.index[len(train)-1], freq='D')
    sarimax_ts = pd.Series(preds, index=sarimax_dates)
    
    # grafico del modello ricavato
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Modello SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = train.plot(label='Train set', color='black')
    sarimax_ts.plot(ax=ax, label='In-sample predictions', color='green')
    plt.legend()
    plt.show()
    
    # out-of-sample forecasts
    # http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.predict
    fcast, conf_int = model.predict(n_periods=test.shape[0], return_conf_int=True)
    
    fcast_dates = pd.date_range(start=timeseries.index[len(train)], periods=len(timeseries)-len(train), freq='D')
    ts_fcast = pd.Series(fcast, index=fcast_dates)
    ts_ci_min = pd.Series(conf_int[:, 0], index=fcast_dates)
    ts_ci_max = pd.Series(conf_int[:, 1], index=fcast_dates)
    
    print('Test RMSE: %.4f'%np.sqrt(mean_squared_error(test, fcast)))
    
    # grafico delle previsioni out-of-sample in sovrapposizione con i dati di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='Out-of-sample forecasts', alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo MAE e MSE
    errore = ts_fcast - timeseries
    errore.dropna(inplace=True)
    print('MSE=%.4f'%(errore ** 2).mean())
    print('MAE=%.4f'%(abs(errore)).mean())


# ===================================TBATS===================================
def tbats_model(timeseries, train_length, s, slow=True):
    """
    Estimazione di un modello TBATS

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)
    s : list
        l'array delle stagionalità
    slow : bool
        se False velocizza il processo di estimazione (di default è True)

    Returns
    -------
    None.

    """
    train = timeseries[pd.date_range(start=timeseries.index[0], end=timeseries.index[int(len(timeseries) * train_length)-1], freq='D')]

    # fit the model
    if slow:
        estimator_slow = TBATS(seasonal_periods=s)
        model = estimator_slow.fit(train)
    else:
        estimator = TBATS(
            seasonal_periods=s,
            use_arma_errors=False,  # shall try only models without ARMA
            use_box_cox=False       # will not use Box-Cox
        )
        model = estimator.fit(train)
    
    # summarize fitted model
    print(model.summary())
    
    # in sample prediction
    preds = model.y_hat
    tbats_dates = pd.date_range(start=timeseries.index[0], end=timeseries.index[len(train)-1], freq='D')
    tbats_ts = pd.Series(preds, index=tbats_dates)
    
    # out-of-sample forecasts
    fcast, conf_int = model.forecast(steps=len(timeseries)-len(train), confidence_level=0.95)
    fcast_dates = pd.date_range(start=timeseries.index[len(train)], periods=len(timeseries)-len(train), freq='D')
    ts_fcast = pd.Series(fcast, index=fcast_dates)
    ts_ci_min = pd.Series(conf_int['lower_bound'], index=fcast_dates)
    ts_ci_max = pd.Series(conf_int['upper_bound'], index=fcast_dates)
    
    # model plot
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Modello TBATS per {}'.format(timeseries.name))
    ax = train.plot(label='Train set', color='black')
    tbats_ts.plot(ax=ax, label='In-sample predictions', color='green')
    plt.legend()
    plt.show()
    print('MAE (in sample)', np.mean(np.abs(model.resid)))
    
    # forecasts plot
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con TBATS per {}'.format(timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='Out-of-sample forecasts', alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # MAE/MSE
    errore = ts_fcast - timeseries
    errore.dropna(inplace=True)
    print('MSE=%.4f'%(errore ** 2).mean())
    print('MAE=%.4f'%(abs(errore)).mean())


def tbats_forecasting(timeseries, h, s, slow=True):
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
    slow : bool
        se False velocizza il processo di estimazione (di default è True)

    Returns
    -------
    None.

    """
    # fit the model
    if slow:
        estimator_slow = TBATS(seasonal_periods=s)
        model = estimator_slow.fit(timeseries)
    else:
        estimator = TBATS(
            seasonal_periods=s,
            use_arma_errors=False,  # shall try only models without ARMA
            use_box_cox=False       # will not use Box-Cox
        )
        model = estimator.fit(timeseries)
    
    # summarize fitted model
    print(model.summary())
    
    # forecasts
    fcast, conf_int = model.forecast(steps=h, confidence_level=0.95)
    fcast_dates = pd.date_range(start=timeseries.index[len(timeseries)-1], periods=h+1, freq='D')
    fcast_dates = fcast_dates[1:]
    ts_fcast = pd.Series(fcast, index=fcast_dates)
    ts_ci_min = pd.Series(conf_int['lower_bound'], index=fcast_dates)
    ts_ci_max = pd.Series(conf_int['upper_bound'], index=fcast_dates)
    
    # forecasts plot
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con TBATS per {}'.format(timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='{} forecasted values'.format(h), alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()


# TODO========================================================================
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


def prophet_forecasting(timeseries, h):
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


def accuracy_prophet(timeseries, end_train):
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


# ====================================MAIN====================================
if __name__ == '__main__':
    data = load_data('Dati_Albignasego/Whole period.csv')
    # colonne: MAGLIE, CAMICIE, GONNE, PANTALONI, VESTITI, GIACCHE
    ts = data['MAGLIE']
    
    #for col in data.columns:
        #ts = data[col]
        
    # %% Grafici, ACF/PACF, stazionarietà
    ts_plot(ts)
    
    # plot di autocorrelazione e autocorrelazione parziale:
    acf_pacf(ts)
    
    # stazionarietà:
    test_stationarity(ts) # the test statistic is smaller than the 1% critical values so we can say with 99% confidence that ts is stationary
    
    # %% Stagionalità     
    decomposition = seasonal_decompose(ts, period=365)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Stagionalità ' + ts.name)
    plt.plot(seasonal)    
    plt.show()
    
    # seasonally adjusted data:
    seas_adj = trend + residual
    seas_adj.dropna(inplace=True)
    seas_adj.name = ts.name
    
    # %% SARIMAX (ignoro la stagionalità annuale)
    #sarimax_statsmodels(ts, 0.8, 7)
    sarimax_pmdarima(ts, 0.8, 7)
    
    # %% TBATS
    tbats_model(ts, 0.8, [7, 365.25], slow=False)
    tbats_forecasting(ts, 100, [7, 365.25], slow=False)
    