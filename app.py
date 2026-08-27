"""
GUI QSAR: disegna una molecola in Ketcher e ottieni la predizione
del tuo modello Random Forest (scikit-learn).

Avvio:
    pip install streamlit streamlit-ketcher rdkit-pypi scikit-learn joblib numpy pandas
    streamlit run app.py

Ogni modello deve essere un file .pkl o .joblib salvato con pickle/joblib,
allenato solo su Morgan fingerprint (2048 bit, raggio 2).
"""

import streamlit as st
from streamlit_ketcher import st_ketcher
import numpy as np
import joblib
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

st.set_page_config(page_title="QSAR Predictor", layout="wide")

# ----------------------------------------------------------------------
# 1. CARICAMENTO MODELLI
# ----------------------------------------------------------------------
# 4 modelli sullo stesso target/dataset, con splitting e algoritmo diversi.
# Metti i file .pkl/.joblib in una cartella "models/" accanto a questo script,
# oppure aggiorna i path qui sotto.
MODELS = {
    "Modello 1 - Random split": "models/randomforest_random_split.pkl",
    "Modello 2 - Scaffold split": "models/randomforest_scaffold_split.pkl",
    "Modello 3 - Temporal split": "models/knn_random_split.pkl",
    "Modello 4 - Cluster split": "models/knn_scaffold_split.pkl",
}


@st.cache_resource
def load_model(path: str):
    if not Path(path).exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# ----------------------------------------------------------------------
# 2. CALCOLO FEATURE (solo Morgan FP 2048 bit r=2)
# ----------------------------------------------------------------------
FP_BITS = 2048
FP_RADIUS = 2


def compute_features(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_BITS
    )
    fp_arr = np.zeros((FP_BITS,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

    return fp_arr.reshape(1, -1)


# ----------------------------------------------------------------------
# 3. INTERFACCIA
# ----------------------------------------------------------------------
st.title("🧪 QSAR Predictor")
st.caption("Disegna una molecola in Ketcher e confronta la predizione tra i 4 modelli.")

st.subheader("Scegli il modello")
selected_name = st.selectbox("Modello", list(MODELS.keys()))
model_path = MODELS[selected_name]
model = load_model(model_path)

if model is None:
    st.warning(
        f"⚠️ Nessun modello trovato in `{model_path}`. "
        "Copia il file .pkl/.joblib in `models/` con quel nome (o aggiorna il path in MODELS), "
        "oppure caricalo qui sotto per questa sessione."
    )
    uploaded_model = st.file_uploader(
        f"Carica '{selected_name}' (.pkl / .joblib)", type=["pkl", "joblib"], key=selected_name
    )
    if uploaded_model is not None:
        try:
            model = joblib.load(uploaded_model)
        except Exception:
            uploaded_model.seek(0)
            model = pickle.load(uploaded_model)
        st.success("Modello caricato correttamente per questa sessione.")

col_editor, col_result = st.columns([2, 1])

with col_editor:
    st.subheader("Disegna la molecola")
    smiles = st_ketcher("", height=500)

with col_result:
    st.subheader("Risultato")

    if not smiles:
        st.info("Disegna una struttura e clicca su 'Apply' in Ketcher per generare lo SMILES.")
    else:
        st.code(smiles, language="text")
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.error("SMILES non valido: controlla la struttura disegnata.")
        elif model is None:
            st.warning("Carica prima un modello per ottenere la predizione.")
        else:
            features = compute_features(smiles)
            try:
                prediction = model.predict(features)[0]
                st.metric("Predizione (pIC50 / valore target)", f"{prediction:.3f}")

                # Se il modello lo supporta, mostra anche un intervallo di
                # confidenza approssimato dalla varianza tra gli alberi RF
                if hasattr(model, "estimators_"):
                    tree_preds = np.array(
                        [tree.predict(features)[0] for tree in model.estimators_]
                    )
                    st.caption(
                        f"Deviazione standard tra gli alberi: {tree_preds.std():.3f} "
                        f"(min {tree_preds.min():.3f} · max {tree_preds.max():.3f})"
                    )
            except Exception as e:
                st.error(f"Errore durante la predizione: {e}")
                st.caption(
                    "Controlla che l'ordine e il numero di feature calcolate "
                    "corrispondano a quelle usate in fase di training."
                )

            if mol is not None and st.checkbox("Confronta tutti e 4 i modelli"):
                comparison = {}
                for name, path in MODELS.items():
                    m = load_model(path)
                    if m is None:
                        comparison[name] = "modello non trovato"
                        continue
                    try:
                        comparison[name] = round(float(m.predict(features)[0]), 3)
                    except Exception as e:
                        comparison[name] = f"errore: {e}"
                st.table(comparison)
