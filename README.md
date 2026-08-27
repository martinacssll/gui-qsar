# QSAR GUI (Ketcher + Random Forest / KNN)

GUI in Streamlit: draw a molecule, app computes
Morgan fingerprint (2048 bit, radius 3) and passes it to the selected model to compute the prediction (pIC50).

## Setup

```bash
pip install -r requirements.txt
```

## Connect the 4 models

The app allows to choose between 4 models (same dataset,
different splitting and model), contained in the folder `models/`.

A dropdown menu allows to choose which model to use to make the prediction.
If a file is missing, the app allows to upload it for the current session.
A checkbox "Compare all 4 models" shows in a table the prediction of each model
on the same molecule.

## Usage

```bash
streamlit run app.py
```

It opens in the browser: draw the molecule, press **Apply** in
Ketcher to confirm the structure, and the prediction is shown on the right.

## Features used by the models

The function `compute\_features()` in `app.py` generates Morgan
fingerprints (2048 bit, radius 3).


