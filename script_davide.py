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


def iterative_order_selection(timeseries, min_order=0, max_order=5):
    """
    Selezione iterativa degli ordini (p,d,q) per ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    min_order : int
        l'ordine minimo da considerare (default 0)
    max_order : int
        l'ordine massimo da considerare (default 5)

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


def arima_model(timeseries, seasonal=pd.Series()):
    """
    Miglior modello ARIMA (basato su una selezione iterativa degli ordini)

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    seasonal: Series
        l'eventuale componente stagionale
    
    Returns
    -------
    tuple
        il modello ARIMA trovato completo di serie temporale e ordini

    """
    # realizzo il modello con gli ordini ottenuti dalla ricerca iterativa
    o = iterative_order_selection(timeseries)
    mod = ARIMA(timeseries, order=o)
    results = mod.fit()
    print(results.summary())
    arima_model = pd.Series(results.fittedvalues, copy=True)
    
    # grafico dei valori osservati e del modello ARIMA
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('ARIMA{} per {}'.format(o, timeseries.name))
    plt.plot(timeseries, label='Seasonally adjusted data', color='black')
    plt.plot(arima_model, label='ARIMA{}'.format(o), color='green')
    plt.legend(loc='best');
    plt.show()
    
    if(not seasonal.empty):
        # unisco al modello trovato la componente stagionale
        arima_model = arima_model + seasonal
        arima_model.dropna(inplace=True)
        observed = timeseries + seasonal
        observed.dropna(inplace=True)
        
        # grafico dei valori osservati e del modello ARIMA (con stagionalità)
        plt.figure(figsize=(40, 20), dpi=80)
        plt.title('ARIMA{} per {}'.format(o, timeseries.name))
        plt.plot(observed, label='Observed', color='black')
        plt.plot(arima_model, label='ARIMA{}'.format(o), color='green')
        plt.legend(loc='best');
        plt.show()

    return results, arima_model, o


def arima_forecasting(timeseries, seasonal, m, mod_arima):
    """
    Forecasting con ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    seasonal : Series
        la componente stagionale
    m : int
        numero di osservazioni in un anno
    mod_arima : tuple
        il modello ARIMA completo di serie temporale e ordini
    
    Returns
    -------
    Series
        la serie dei valori predetti
        
    """
    results = mod_arima[0]
    model = mod_arima[1]
    orders = mod_arima[2]
    h = len(timeseries) - len(seasonal)
    last_observation = model.index[len(model) - 1]        
    
    # previsioni sulla componente stagionale
    ts_seasonal_forecast = pd.Series(seasonal[model.index], copy=True)
    sfasamento = int((len(seasonal) - len(model))/2)
    
    tmp = [0.0] * h                     # conterrà i valori di previsione stagionale, dati dalla media dei valori dello stesso periodo
    start = len(ts_seasonal_forecast)   # rappresenta l'osservazione futura da prevedere
    for i in range(0, h):               # 0 sarebbe t+1 e arriva a t+1+h-1=t+h
        ind = start
        alpha = 0.9                     # sommatoria in media exp
        ind -= m                        # prima il decremento perchè non abbiamo il valore di t+1 
        tmp[i] += seasonal[sfasamento + ind]
        exp = 1
        while ind >= 0:
            tmp[i] += seasonal[sfasamento + ind] * ((1 - alpha) ** exp)
            exp += 1
            ind -= m                     # prima il decremento perchè non abbiamo il valore di t+1 
        
        start += 1                      # questo arriverà fino a t+h
        tmp[i] = tmp[i]
    
    ts_seasonal_forecast_h = pd.Series(data=tmp, index=pd.date_range(start=last_observation, periods=h , freq='D'))
    ts_seasonal_forecast = ts_seasonal_forecast.add(ts_seasonal_forecast_h, fill_value=0)
    
    tmp = [0.0] * sfasamento            # conterrà i valori di previsione stagionale, dati dalla media dei valori dello stesso periodo
    start = len(ts_seasonal_forecast)   # rappresenta l'osservazione futura da prevedere
    for i in range(0, sfasamento):      # 0 sarebbe t+1 e arriva a t+1+h-1=t+h
        ind = start
        alpha = 0.9                     # sommatoria in media exp
        ind -= m                        # prima il decremento perchè non abbiamo il valore di t+1 
        tmp[i] += ts_seasonal_forecast[ind]
        exp = 1
        while ind >= 0:
            tmp[i] += ts_seasonal_forecast[ind] * ((1 - alpha) ** exp)
            exp += 1
            ind -= m                    # prima il decremento perchè non abbiamo il valore di t+1 
        
        start += 1                      # questo arriverà fino a t+h
        tmp[i] = tmp[i]
    
    ts_seasonal_forecast_h = pd.Series(data=tmp, index=pd.date_range(start=ts_seasonal_forecast.index[len(ts_seasonal_forecast) - 1], periods=sfasamento, freq='D'))
    ts_seasonal_forecast = ts_seasonal_forecast.add(ts_seasonal_forecast_h, fill_value=0)
    
    # previsioni sulla parte de-stagionata
    new_h = h + sfasamento
    (previsione, _ ,intervallo) = results.forecast(steps=new_h)
    ts_NOseasonal_forecast = pd.Series(previsione, index=pd.date_range(start=last_observation, periods=new_h, freq='D'))
    
    # previsioni totali
    ts_forecast = ts_seasonal_forecast + ts_NOseasonal_forecast
    
    # intervalli
    intervallo_sup = [0.0] * new_h
    intervallo_inf = [0.0] * new_h
    seasonal_interval_sum = [0.0] * new_h
    ind = 0
    for n in intervallo[:, [0]]:
        intervallo_sup[ind] = float(n)
        ind+=1
    ind = 0
    for n in intervallo[:, [1]]:
        intervallo_inf[ind] = float(n)
        ind+=1
    
    ind = 0
    for i in range(len(ts_seasonal_forecast) - new_h, len(ts_seasonal_forecast)):
        seasonal_interval_sum[ind] = float(ts_seasonal_forecast[i])
        ind+=1
    
    for i in range(0, new_h):
        intervallo_sup[i] += seasonal_interval_sum[i]
    for i in range(0, new_h):
        intervallo_inf[i] += seasonal_interval_sum[i]
    
    # grafico
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con ARIMA{} per {}'.format(orders, timeseries.name))
    plt.plot(timeseries, label='Observed', color='black')
    plt.plot(model, label='ARIMA{} model'.format(orders), color='green')
    plt.plot(ts_forecast, label='Forecast', color='red')
    plt.fill_between(pd.date_range(start=last_observation, periods=new_h , freq='D'), 
                 intervallo_sup, 
                 intervallo_inf, 
                 color='k', alpha=.25)
    plt.legend(loc='best');
    plt.show()
    
    return ts_forecast


def accuracy_arima(timeseries, train_length, m):
    """
    Verifica dell'accuratezza di un modello ARIMA

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)
    m : int
        numero di osservazioni in un anno

    Returns
    -------
    None.

    """
    # spezzo la serie temporale
    train = timeseries[pd.date_range(start=data.index[0], end=timeseries.index[int(len(timeseries) * train_length)], freq='D')]

    decomposition = seasonal_decompose(train, period=m)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    seas_adj = trend + residual
    seas_adj.dropna(inplace=True)
    seas_adj.name = ts.name
    
    # cerco un modello
    mod_arima = arima_model(seas_adj, seasonal=seasonal)

    # predico valori
    test_forecast = arima_forecasting(timeseries, seasonal, m, mod_arima)
    
    # calcolo MSE
    test_forecast.dropna(inplace=True)
    test = timeseries[pd.date_range(start=seas_adj.index[len(seas_adj)-1], end=timeseries.index[len(timeseries)-1], freq='D')]
    se = test_forecast - test
    se.dropna(inplace=True)
    
    print("MSE=%.4f"%(se ** 2).mean())
    
    # calcolo MAE e MAPE
    errore = test_forecast - timeseries
    errore.dropna(inplace=True)
    
    sommaPrevOss = test_forecast + timeseries
    sommaPrevOss.dropna(inplace=True)
    
    print("MAE=%.4f"%(sum(abs(errore))/len(errore)))
    print("MAPE=%.4f"%(sum(200 * abs(errore) / sommaPrevOss)/len(sommaPrevOss)))


def sarimax_forecasting(timeseries, m, h):
    """
    Forecasting con SARIMAX

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    m : int
        stagionalità
    h : int
        l'orizzonte

    Returns
    -------
    PredictionResultsWrapper
        le informazioni sui risultati del forecast sul modello SARIMAX

    """
    # realizzo il modello con gli ordini ottenuti da pmdarima
    model = pm.auto_arima(timeseries, seasonal=True, m=m, suppress_warnings=True, trace=True)
    print(model.summary())
    
    # controllo la sua bontà
    plt.figure(figsize=(40, 20), dpi=80)
    model.plot_diagnostics(figsize=(40, 20))
    plt.show()
    
    # predico h valori
    forecast = model.arima_res_.get_forecast(steps=h)
    
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    forecast_dates = pd.date_range(start=timeseries.index[len(timeseries)-1], periods=h+1, freq='D')
    forecast_dates = forecast_dates[1:]
    ts_forecast = pd.Series(forecast_values, index=forecast_dates)
    ts_ci_min = pd.Series(forecast_ci[:, 0], index=forecast_dates)
    ts_ci_max = pd.Series(forecast_ci[:, 1], index=forecast_dates)
    
    # grafico dei valori osservati e dei valori predetti
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_forecast.plot(ax=ax, label='Forecast', color='red')
    ax.fill_between(forecast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.25)
    plt.legend()
    plt.show()
    
    return forecast


def accuracy_sarimax(timeseries, train_length, m):
    """
    Verifica dell'accuratezza di un modello SARIMAX

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
    train = timeseries[pd.date_range(start=data.index[0], end=timeseries.index[int(len(timeseries) * train_length)], freq='D')]

    # realizzo il modello sulla serie completa
    model = pm.auto_arima(timeseries, seasonal=True, m=m, suppress_warnings=True, trace=True)
    print(model.summary())
    
    # controllo la sua bontà
    plt.figure(figsize=(40, 20), dpi=80)
    model.plot_diagnostics(figsize=(40, 20))
    plt.show()
    
    # osservo i valori predetti
    pred = model.arima_res_.get_prediction(start=len(train), dynamic=False)
    pred_ci = pred.conf_int()
    
    pred_dates = pd.date_range(start=timeseries.index[int(len(timeseries) * train_length)+1], periods=len(timeseries)-len(train), freq='D')
    ts_pred = pd.Series(pred.predicted_mean, index=pred_dates)
    ts_ci_min = pd.Series(pred_ci[:, 0], index=pred_dates)
    ts_ci_max = pd.Series(pred_ci[:, 1], index=pred_dates)
    
    # ricavo il modello
    sarimax_mod = model.arima_res_.get_prediction(end=len(train)-1, dynamic=False)
    sarimax_dates = pd.date_range(start=data.index[0], end=timeseries.index[len(train)-1], freq='D')
    sarimax_ts = pd.Series(sarimax_mod.predicted_mean, index=sarimax_dates)
    
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    sarimax_ts.plot(ax=ax, label='SARIMAX{}x{} model'.format(model.order, model.seasonal_order), color='green')
    ts_pred.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, color='red')
    ax.fill_between(pred_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo MSE
    se = ts_pred - timeseries
    se.dropna(inplace=True)
    
    print("MSE=%.4f"%(se ** 2).mean())
    
    # calcolo MAE e MAPE
    errore = ts_pred - timeseries
    errore.dropna(inplace=True)
    
    sommaPrevOss = ts_pred + timeseries
    sommaPrevOss.dropna(inplace=True)
    
    print("MAE=%.4f"%(sum(abs(errore))/len(errore)))
    print("MAPE=%.4f"%(sum(200 * abs(errore) / sommaPrevOss)/len(sommaPrevOss)))


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
        use_box_cox=False       # will not use Box-Cox
    )
    model = estimator.fit(timeseries)
    
    # fit the model (slow)
    #estimator_slow = TBATS(seasonal_periods=s)
    #model = estimator_slow.fit(timeseries)
    
    # summarize fitted model
    print(model.summary())
    
    y_forecasted, confidence_info = model.forecast(steps=h, confidence_level=0.95)
    
    return (y_forecasted, confidence_info)


def accuracy_tbats(timeseries, train_length):
    """
    Verifica dell'accuratezza di un modello TBATS

    Parameters
    ----------
    timeseries : Series
        la serie temporale
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa)

    Returns
    -------
    None.

    """
    # spezzo la serie temporale
    train = timeseries[pd.date_range(start=data.index[0], end=timeseries.index[int(len(timeseries) * train_length)], freq='D')]

    # forecasting con TBATS:
    (forecast, forecast_ci) = tbats_forecasting(train, len(ts)-len(train), [7, 365.25])
    
    # calcolo la serie temporale dei valori predetti:
    future_dates = pd.date_range(start=ts.index[len(train)], periods=len(timeseries)-len(train), freq='D')
    future_series = pd.Series(data=forecast, index=future_dates)
    ci_min = pd.Series(forecast_ci['lower_bound'], index=future_dates)
    ci_max = pd.Series(forecast_ci['upper_bound'], index=future_dates)
    
    # grafico dei valori predetti in sovrapposizione con quelli del set di test
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con TBATS per {}'.format(timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    future_series.plot(ax=ax, label='Forecast', alpha=.7, color='red')
    ax.fill_between(future_dates, ci_min, ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # calcolo MSE
    se = future_series - timeseries
    se.dropna(inplace=True)
    print("MSE=%.4f"%(se ** 2).mean())
    
    # calcolo MAE
    test = timeseries[pd.date_range(start=train.index[len(train)-1], end=timeseries.index[len(timeseries)-1], freq='D')]
    test = test[1:]
    print("MAE=%.4f"%np.mean(np.abs(forecast - test)))


# TODO
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
    
    # %% ARIMA
    # modello ARIMA (basato sugli orders ottenuti minimizzando AIC):
    mod_arima = arima_model(seas_adj, seasonal=seasonal)
    
    # forecasting con ARIMA:
    fcast_arima = arima_forecasting(ts, seasonal, 365, mod_arima)
    
    # controllo l'accuratezza delle previsioni di ARIMA confrontandole con la serie stessa:
    accuracy_arima(ts, 0.8, 365)
    
    # %% SARIMAX
    # forecasting con SARIMAX (basato sugli orders ottenuti da pmdarima):
    fcast_sarimax = sarimax_forecasting(ts, 7, 50) # ignoro la stagionalità annuale
    
    # controllo l'accuratezza delle previsioni di SARIMAX confrontandole con la serie stessa:
    accuracy_sarimax(ts, 0.8, 7)
    
    # %% TBATS
    # controllo l'accuratezza delle previsioni di TBATS confrontandole con la serie stessa:
    accuracy_tbats(ts, 0.8)
    