"""
GUI QSAR: draw a molecule in Ketcher and get a prediction
from different QSAR models

Usage:
    pip install streamlit streamlit-ketcher rdkit-pypi scikit-learn joblib numpy pandas
    streamlit run app.py

Each model must be a .pkl or .joblib file saved with pickle/joblib,
trained only on Morgan fingerprint (2048 bit, radius 2).
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
# 1. MODELS UPLOAD
# ----------------------------------------------------------------------
# 4 models on the same dataset, with different splitting and algorithms

MODELS = {
    "Model 1 - Random split, Random Forest": "models/randomforest_random_split.pkl",
    "Model 2 - Scaffold split, Random Forest": "models/randomforest_scaffold_split.pkl",
    "Model 3 - Random split, K-Nearest Neighbors": "models/knn_random_split.pkl",
    "Model 4 - Scaffold split, K-Nearest Neighbors": "models/knn_scaffold_split.pkl",
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
# 2. cOMPUTE FEATURE (Morgan FP 2048 bit r=2)
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
# 3. GUI
# ----------------------------------------------------------------------
st.title("🧪 QSAR Predictor")
st.caption("Draw a molecule in Ketcher and compare the prediction obtained from 4 models")

st.subheader("Choose model")
selected_name = st.selectbox("Model", list(MODELS.keys()))
model_path = MODELS[selected_name]
model = load_model(model_path)

if model is None:
    st.warning(
        f"⚠️ No model found in `{model_path}`. "
        "Copy the .pkl/.joblib file in `models/` with that name (or update the path in MODELS), "
        "or upload it here for the current session."
    )
    uploaded_model = st.file_uploader(
        f"Upload '{selected_name}' (.pkl / .joblib)", type=["pkl", "joblib"], key=selected_name
    )
    if uploaded_model is not None:
        try:
            model = joblib.load(uploaded_model)
        except Exception:
            uploaded_model.seek(0)
            model = pickle.load(uploaded_model)
        st.success("Model uploaded successfully for the current session.")

col_editor, col_result = st.columns([2, 1])

with col_editor:
    st.subheader("Draw the molecule")
    smiles = st_ketcher("", height=500)

with col_result:
    st.subheader("Result")

    if not smiles:
        st.info("Draw a structure and click 'Apply' in Ketcher to generate the SMILES.")
    else:
        st.code(smiles, language="text")
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.error("Invalid SMILES: check the structure.")
        elif model is None:
            st.warning("Upload a model first to get a prediction.")
        else:
            features = compute_features(smiles)
            try:
                prediction = model.predict(features)[0]
                st.metric("Prediction (pIC50)", f"{prediction:.3f}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.caption(
                    "Ensure that the order and number of calculated features "
                    "correspond to those used during training."
                )

            if mol is not None and st.checkbox("Compare all 4 models"):
                comparison = {}
                for name, path in MODELS.items():
                    m = load_model(path)
                    if m is None:
                        comparison[name] = "model not found"
                        continue
                    try:
                        comparison[name] = round(float(m.predict(features)[0]), 3)
                    except Exception as e:
                        comparison[name] = f"error: {e}"
                st.table(comparison)
