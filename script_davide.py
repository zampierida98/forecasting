# -*- coding: utf-8 -*-
"""
Analisi dei dati di un negozio di abbigliamento
"""
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import statsmodels.api as sm
import datetime as dt

# %% Definizione funzioni


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
    plt.title('Autocorrelation Function')
    # plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
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


# %% Caricamento dei dati
dateparse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('Dati_Albignasego/Whole period.csv', index_col=0, date_parser=dateparse)
print(data)

ts_maglie = data['MAGLIE']
ts_camicie = data['CAMICIE']
ts_gonne = data['GONNE']
ts_pantaloni = data['PANTALONI']
ts_vestiti = data['VESTITI']
ts_giacche = data['GIACCHE']
ts = ts_maglie + ts_camicie + ts_gonne + ts_pantaloni + ts_vestiti + ts_giacche

ts.plot(figsize=(40,20), title= 'Dati_Albignasego', fontsize=14)


# %% Ricerca di stagionalità e stazionarietà; plot di autocorrelazione e autocorrelazione parziale
decomposition = seasonal_decompose(ts, period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

seasonal.plot(figsize=(40,20), title= 'Stagionalità Dati_Albignasego', fontsize=14)

test_stationarity(ts) # the test statistic is smaller than the 1% critical values so we can say with 99% confidence that ts is stationary

acf_pacf(ts)


# %% Forecasting con pmdarima
[o, so] = order_selection(ts)
mod = sm.tsa.statespace.SARIMAX(ts,
                                order=o,
                                seasonal_order=so,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary())

results.plot_diagnostics(figsize=(40,20))
plt.show()

pred_uc = results.get_forecast(steps=24)
pred_ci = pred_uc.conf_int()

ax = ts.plot(label='observed', figsize=(40,20))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
plt.legend()
plt.show()

print(pred_uc.predicted_mean)
