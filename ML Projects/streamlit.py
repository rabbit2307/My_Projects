import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Celestial Object Classifier", layout="centered")

MODEL_PATH = 'xgb_sdss_classifier.pkl'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the same directory.")
    st.stop()
if not os.path.exists(SCALER_PATH):
    st.error(f"Error: Scaler file not found at {SCALER_PATH}. Please ensure it's in the same directory.")
    st.stop()
if not os.path.exists(LABEL_ENCODER_PATH):
    st.error(f"Error: Label Encoder file not found at {LABEL_ENCODER_PATH}. Please ensure it's in the same directory.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    st.success("Model, scaler, and label encoder loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or preprocessors: {e}")
    st.stop()

feature_names = [
    'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol',
    'field', 'redshift', 'plate', 'mjd', 'fiberid'
]

def classify_celestial_object(input_data_dict):
    input_df = pd.DataFrame([input_data_dict], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction_encoded = model.predict(input_scaled)
    predicted_class = le.inverse_transform(prediction_encoded)[0]
    return predicted_class


st.title("Celestial Object Classifier")
st.markdown("""
    Enter the astronomical parameters below to classify a celestial object as a
    **Galaxy**, **Quasar (QSO)**, or **Star**.
""")

st.header("Object Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ra = st.number_input("Right Ascension (ra)", value=183.680207, format="%.6f")
    u = st.number_input("u-band magnitude (u)", value=19.38298, format="%.5f")
    r = st.number_input("r-band magnitude (r)", value=17.47428, format="%.5f")
    z = st.number_input("z-band magnitude (z)", value=16.80125, format="%.5f")
    rerun = st.number_input("Rerun (rerun)", value=301, step=1)

with col2:
    dec = st.number_input("Declination (dec)", value=0.126185, format="%.6f")
    g = st.number_input("g-band magnitude (g)", value=18.19169, format="%.5f")
    i = st.number_input("i-band magnitude (i)", value=17.08732, format="%.5f")
    run = st.number_input("Run (run)", value=752, step=1)
    camcol = st.number_input("Camcol (camcol)", value=4, step=1)

with col3:
    redshift = st.number_input("Redshift (redshift)", value=0.123111, format="%.6f")
    plate = st.number_input("Plate (plate)", value=287, step=1)
    mjd = st.number_input("MJD (mjd)", value=52023, step=1)
    fiberid = st.number_input("Fiber ID (fiberid)", value=513, step=1)
    field = st.number_input("Field (field)", value=268, step=1)


input_data = {
    'ra': ra, 'dec': dec, 'u': u, 'g': g, 'r': r, 'i': i, 'z': z,
    'run': run, 'rerun': rerun, 'camcol': camcol, 'field': field,
    'redshift': redshift, 'plate': plate, 'mjd': mjd, 'fiberid': fiberid
}

if st.button("Classify Object"):
    with st.spinner("Classifying..."):
        predicted_class = classify_celestial_object(input_data)
        st.success(f"The predicted class for the object is: **{predicted_class}**")

st.markdown("---")
st.markdown("Built with using Streamlit and Machine Learning models.")
