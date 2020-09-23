# -*- coding: utf-8 -*-
"""
Analisi dei dati di un negozio di abbigliamento
"""
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tbats import TBATS

# imposto i parametri comuni a tutti i grafici
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
	
	# preparo un indice di tipo datetime
    dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
	
	# leggo i dati nel file csv
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
	
	# creo il grafico della serie temporale
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
	
	# test Dickey-Fuller
    print('Results of Dickey-Fuller Test for', timeseries.name, '\n')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def acf_pacf(timeseries, lags=40):
    """
    Realizza i grafici di autocorrelazione e autocorrelazione parziale

    Parameters
    ----------
    timeseries : Series
        la serie temporale.
    lags : int
        il numero di osservazioni desiderate.

    Returns
    -------
    None.

    """
    lag_acf = acf(timeseries, nlags=lags, fft=True)
    lag_pacf = pacf(timeseries, nlags=lags, method='ols')
    plt.figure(figsize=(40, 20), dpi=80)
	
    # ACF
    plt.subplot(211)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.title('Autocorrelation Function '+timeseries.name)
    
    # PACF
    plt.subplot(212)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='red')
    plt.title('Partial Autocorrelation Function '+timeseries.name)
    plt.tight_layout()
    plt.show()


# ==================================SARIMAX==================================
def sarimax_pmdarima(timeseries, train_length, m):
    """
    Previsioni con il modello SARIMAX e selezione automatica degli ordini

    Parameters
    ----------
    timeseries : Series
        la serie temporale.
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa).
    m : int
        il periodo stagionale.

    Returns
    -------
    tuple
        (order, seasonal_order)

    """
    # creo i set di train e di test
    train, test = model_selection.train_test_split(timeseries, train_size=train_length)

    # scelgo e adatto il modello ai dati
    model = pm.auto_arima(train, seasonal=True, m=m, suppress_warnings=True, trace=True,
                          start_p=1, start_q=1, max_p=2, max_q=2, start_P=1, start_Q=1, max_P=2, max_Q=2)
    
	# stampo i parametri del modello
	print(model.summary())
    
    # predizioni in-sample
    # http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.predict_in_sample
    preds = model.predict_in_sample(end=len(train)-1)
    sarimax_dates = pd.date_range(start=timeseries.index[0], end=timeseries.index[len(train)-1], freq='D')
    sarimax_ts = pd.Series(preds, index=sarimax_dates)
    
    # predizioni out-of-sample
    # http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.predict
    fcast, conf_int = model.predict(n_periods=test.shape[0], return_conf_int=True)
    fcast_dates = pd.date_range(start=timeseries.index[len(train)], periods=len(timeseries)-len(train), freq='D')
    ts_fcast = pd.Series(fcast, index=fcast_dates)
    ts_ci_min = pd.Series(conf_int[:, 0], index=fcast_dates)
    ts_ci_max = pd.Series(conf_int[:, 1], index=fcast_dates)
    
    print('Test RMSE: %.4f'%np.sqrt(mean_squared_error(test, fcast)))
    
    # grafico del modello
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Modello SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = train.plot(label='Train set', color='black')
    sarimax_ts.plot(ax=ax, label='In-sample predictions', color='green')
    plt.legend()
    plt.show()
    
    # grafico delle previsioni
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con SARIMAX{}x{} per {}'.format(model.order, model.seasonal_order, timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='Out-of-sample forecasts', alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # metriche di errore
    errore = ts_fcast - timeseries
    errore.dropna(inplace=True)
    print('MSE=%.4f'%(errore ** 2).mean())
    print('MAE=%.4f'%(abs(errore)).mean())
    
    return (model.order, model.seasonal_order)


def sarimax_statsmodels(timeseries, train_length, o, so):
    """
    Previsioni con il modello SARIMAX

    Parameters
    ----------
    timeseries : Series
        la serie temporale.
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa).
    o : iterable
        order del modello SARIMAX (per statsmodels).
    so : iterable
        seasonal_order del modello SARIMAX (per statsmodels).

    Returns
    -------
    None.

    """
	
	# controllo se i dati sono settimanali o giornalieri
    if so[3] == 52:
        f = 'W-MON'
    else:
        f = 'D'
    
    # creo il set di train
    train = timeseries[pd.date_range(start=timeseries.index[0], end=timeseries.index[int(len(timeseries) * train_length)-1], freq=f)]

    # adatto il modello ai dati
    model = smt.SARIMAX(train, order=o, seasonal_order=so, trend='c').fit()
    #model = pm.auto_arima(train, seasonal=True, m=m, suppress_warnings=True, trace=True,
                          #start_p=1, start_q=1, max_p=1, max_q=1, start_P=1, start_Q=1, max_P=1, max_Q=1)
    
	# stampo i parametri del modello e controllo la sua bontà
	print(model.summary())
    plt.figure(figsize=(40, 20), dpi=80)
    model.plot_diagnostics(figsize=(40, 20))
    plt.show()
    
    # predizioni in-sample
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_prediction.html
    sarimax_mod = model.get_prediction(end=len(train)-1, dynamic=False)
    sarimax_dates = pd.date_range(start=timeseries.index[0], end=timeseries.index[len(train)-1], freq=f)
    sarimax_ts = pd.Series(sarimax_mod.predicted_mean, index=sarimax_dates)
    
    # predizioni out-of-sample
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast.html
    fcast = model.get_forecast(steps=len(timeseries)-len(train))
    fcast_ci = fcast.conf_int()
    fcast_dates = pd.date_range(start=timeseries.index[len(train)], periods=len(timeseries)-len(train), freq=f)
    ts_fcast = pd.Series(fcast.predicted_mean, index=fcast_dates)
    
    # grafico del modello
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Modello SARIMAX{}x{} per {}'.format(o, so, timeseries.name))
    ax = train.plot(label='Train set', color='black')
    sarimax_ts.plot(ax=ax, label='In-sample predictions', color='green')
    plt.legend()
    plt.show()
    
    # grafico delle previsioni
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con SARIMAX{}x{} per {}'.format(o, so, timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='Out-of-sample forecasts', alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    fcast_ci['lower '+timeseries.name],
                    fcast_ci['upper '+timeseries.name], color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # metriche di errore
    errore = ts_fcast - timeseries
    errore.dropna(inplace=True)
    print('MSE=%.4f'%(errore ** 2).mean())
    print('MAE=%.4f'%(abs(errore)).mean())


# ===================================TBATS===================================
def tbats_model(timeseries, train_length, s, slow=True):
    """
    Previsioni con il modello TBATS

    Parameters
    ----------
    timeseries : Series
        la serie temporale.
    train_length : int
        la lunghezza del set di train (in rapporto alla serie completa).
    s : list
        l'array dei periodi stagionali.
    slow : bool
        se False velocizza il processo di scelta del modello finale (di default è True).

    Returns
    -------
    None.

    """
	
	# controllo se i dati sono settimanali o giornalieri
    if s.count(52) == 1:
        f = 'W-MON'
    else:
        f = 'D'
    
	# creo il set di train
    train = timeseries[pd.date_range(start=timeseries.index[0], end=timeseries.index[int(len(timeseries) * train_length)-1], freq=f)]

    # adatto il modello ai dati
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
    
    # stampo i parametri del modello
    print(model.summary())
    
    # predizioni in-sample (model.y_hat = train - model.resid)
    preds = model.y_hat
    tbats_dates = pd.date_range(start=timeseries.index[0], end=timeseries.index[len(train)-1], freq=f)
    tbats_ts = pd.Series(preds, index=tbats_dates)
    
    # predizioni out-of-sample
    fcast, conf_int = model.forecast(steps=len(timeseries)-len(train), confidence_level=0.95)
    fcast_dates = pd.date_range(start=timeseries.index[len(train)], periods=len(timeseries)-len(train), freq=f)
    ts_fcast = pd.Series(fcast, index=fcast_dates)
    ts_ci_min = pd.Series(conf_int['lower_bound'], index=fcast_dates)
    ts_ci_max = pd.Series(conf_int['upper_bound'], index=fcast_dates)
    
    # grafico del modello
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Modello TBATS per {}'.format(timeseries.name))
    ax = train.plot(label='Train set', color='black')
    tbats_ts.plot(ax=ax, label='In-sample predictions', color='green')
    plt.legend()
    plt.show()
    print('MAE (in sample)', np.mean(np.abs(model.resid)))
    
    # grafico delle previsioni
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Forecasting con TBATS per {}'.format(timeseries.name))
    ax = timeseries.plot(label='Observed', color='black')
    ts_fcast.plot(ax=ax, label='Out-of-sample forecasts', alpha=.7, color='red')
    ax.fill_between(fcast_dates,
                    ts_ci_min,
                    ts_ci_max, color='k', alpha=.2)
    plt.legend()
    plt.show()
    
    # metriche di errore
    errore = ts_fcast - timeseries
    errore.dropna(inplace=True)
    print('MSE=%.4f'%(errore ** 2).mean())
    print('MAE=%.4f'%(abs(errore)).mean())


# ====================================MAIN====================================
if __name__ == '__main__':
	# colonne: MAGLIE, CAMICIE, GONNE, PANTALONI, VESTITI, GIACCHE
    data = load_data('Dati_Albignasego/Whole period.csv')
	
	# analizzo la serie temporale delle MAGLIE
	ts = data['MAGLIE']
    
	# per le altre serie il procedimento è analogo
    #for col in data.columns:
        #ts = data[col]
        
    # %% Grafici, ACF/PACF, stazionarietà:
	# grafico della serie temporale
    ts_plot(ts)
    
    # grafici di autocorrelazione e autocorrelazione parziale
    acf_pacf(ts)
    
    # test Dickey-Fuller (stazionarietà)
    test_stationarity(ts) # the test statistic is smaller than the 1% critical values so we can say with 99% confidence that ts is stationary
    
    # %% Stagionalità:
	# decomposizione
    decomposition = seasonal_decompose(ts, period=365)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
	# grafico della componente stagionale
    plt.figure(figsize=(40, 20), dpi=80)
    plt.title('Stagionalità ' + ts.name)
    plt.plot(seasonal)    
    plt.show()
    
    # %% SARIMAX:
	# modello SARIMAX con stagionalità settimanale (ignoro quella semestrale e annuale)
    (o, so) = sarimax_pmdarima(ts, 0.8, 7)
    
    # modello SARIMAX con stagionalità annuale (365 giorni)
    new_so = tuple()
    for i in range(0,len(so)-1):
        new_so += (so[i],)
    new_so += (365,)
    #sarimax_statsmodels(ts, 0.8, o, new_so) # errore di memoria
    
    # %% TBATS:
	# modello TBATS con stagionalità settimanale, annuale e semestrale
    tbats_model(ts, 0.8, [7, 365.25, 182.625], slow=False)
       
    # %% Aggregazione settimanale dei dati tramite media:
    # aggiusto la serie togliendo le settimane incomplete all'inizio e alla fine
    adj_ts = ts[2:] # 25-03-2013 è lunedì, 29-09-2019 è domenica
    new_dates = pd.date_range(start=adj_ts.index[0], periods=len(adj_ts)/7, freq='W-MON')
	
	# aggrego i dati (tramite media)
    new_data = []
    for week in range(0, len(adj_ts), 7):
        somma = 0
        for day in range(0,7):
            somma += adj_ts[week+day]
        new_data.append(somma/7)
        
	# creo la serie temporale con i dati aggregati
    new_ts = pd.Series(data=new_data, index=new_dates)
    new_ts.name = ts.name + ' (aggregazione settimanale)'
    
	# modello SARIMAX con stagionalità annuale (52 settimane)
    new_so = tuple()
    for i in range(0,len(so)-1):
        new_so += (so[i],)
    new_so += (52,)
    sarimax_statsmodels(new_ts, 0.8, o, new_so)
    
    # modello TBATS con stagionalità annuale e semestrale (52 e 26 settimane)
    tbats_model(new_ts, 0.8, [52, 26], slow=False)
    