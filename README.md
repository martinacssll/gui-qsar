# QSAR Predictor (Ketcher + Random Forest)

GUI in Streamlit: disegni una molecola in Ketcher, l'app calcola il
Morgan fingerprint (2048 bit, raggio 2) e lo passa al modello scelto per ottenere la predizione.

## Setup

```bash
pip install -r requirements.txt
```

## Collegare i tuoi 4 modelli

L'app permette di scegliere tra 4 modelli (stesso target/dataset,
diverso splitting e modello). Crea una cartella `models/` accanto a `app.py` e
mettici i 4 file, con questi nomi (o cambia i path nel dizionario
`MODELS` in cima ad `app.py`):

```
models/model\_random\_split.pkl
models/model\_scaffold\_split.pkl
models/model\_temporal\_split.pkl
models/model\_cluster\_split.pkl
```

Un menu a tendina permette di scegliere quale usare per la predizione.
Se un file manca, l'app mostra un uploader per caricarlo per quella
sessione. C'è anche una checkbox "Confronta tutti e 4 i modelli" che
mostra in una tabella la predizione di ciascuno sulla stessa molecola.

## Avvio

```bash
streamlit run app.py
```

Si apre nel browser: disegni la molecola, premi **Apply** dentro
Ketcher per confermare la struttura, e la predizione appare a destra.

## Se il tuo modello usa feature diverse

La funzione `compute\_features()` in `app.py` genera solo il Morgan
fingerprint (2048 bit, raggio 2). Se in futuro un modello usasse feature
aggiuntive, modifica quella funzione di conseguenza — l'ordine delle
colonne deve corrispondere esattamente a quello usato per allenare il
modello, altrimenti la predizione sarà silenziosamente sbagliata.



