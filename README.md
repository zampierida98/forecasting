# Analisi Predittive di Serie Storiche
Link al diario: https://docs.google.com/document/d/1_r1HHRjRuGP4bbE47mgb6zxnemk_Q8J1uCu5me7eT2c

## Fasi dello studio dei dati di un negozio di abbigliamento
- Studio della serie e modello ARIMA: `script_sebastiano.py` e analisi simili anche in `script_michele.py` (le previsioni della componente stagionale vengono realizzate usando una media exp con alpha=0.9)
- Modelli di exponential smoothing: `script_ETS.py`
- Modello TBATS per stagionalità multiple e confronto con dati aggregati: `script_davide.py`

## Fasi dello studio dei dati della M5 competition (serie gerarchica)
Link ai dati della M5 competition (da inserire nella directory `M5_COMPETITION/datasets`): https://drive.google.com/file/d/19u1pDaCA_sFQheWdn5qStKuDfuoaFg5b/view?usp=sharing
- Modello ETS con approccio bottom-up e diretto: `m5_competition.py`
- Modello ARIMA con approccio bottom-up e diretto: `m5_competition_ARIMA.py`

## Procedura per l'esecuzione degli script
Per realizzare i programmi è stato usato l'IDE Spyder poiché è un ambiente che mette a disposizione diverse librerie e in cui i grafici vengono visualizzati in maniera ottimale. Descriviamo quindi due procedure per eseguire i programmi dove la prima è la più consigliata.

### Procedura 1
- Scaricare Spyder tramite Anaconda
- Installare attraverso la console di Spyder (sezione in basso a destra) le librerie pmdarima e tbats:
```
pip install pmdarima tbats
```

### Procedura 2
- Scaricare IPython (versione>=7.18)
- Installare le seguenti librerie tramite pip:
	* matplotlib
	* pandas
	* statsmodels
	* pmdarima
	* sklearn
	* tbats
```
pip install matplotlib pandas statsmodels sklearn pmdarima tbats
```

- I programmi vanno lanciati dentro le rispettive directory poiché usiamo path relativi per accedere ai file
