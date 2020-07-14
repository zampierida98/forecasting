# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:38:31 2020

@author: seba3
"""

# ANALISI DEI DATI M5 COMPETITION

# Nella competizione per "accuracy" si usa la seguente metrica
# d'errore: Weighted Root Mean Squared Scaled Error (RMSSE)

"""
Ci viene chiesto di prevedere serie gerarchiche che si riferiscono a Wal-Mart in
3 stati diversi (California, Texas e Wisconsin) e include "item level", categoria prodotto, dipartimento
e dettagli del negozio oltre a prezzo, promozioni, giorno della settimana e eventi speciali.

File di dati:
 -calendar.csv: contiene informazioni sulla data in cui prodotti sono venduti.
 -sales_train_validation.csv: contiene lo storico delle vendite per prodotto e negozio [d_1 - d_1913]
 -sample_submission.csv: il formato con cui fornire la soluzione alla competizione.
 -sell_prices.csv: contiene informazioni sul prezzo dei prodotti venduti per negozi e per data
 -sales_train_evaluation.csv: include le vendite [d_1 - d_1941]
 
Ci viene richiesto di prevedere i prossimi 28 giorni
- le colonne si riferiscono al giorno
- le righe rappresentano un oggetto specifico

Il periodo [d_1 - d_1913] rappresenta il train set
Il periodo [d_1914 - d_1941] rappresenta il validation set
Il periodo [d_1942 - d_1969] rappresenta il set di valutazione
"""