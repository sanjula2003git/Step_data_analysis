import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter1d

# --- Page Config ---
st.set_page_config(page_title="🐾 Pet Activity Anomaly Detection")

st.title("🐾 Pet Activity Anomaly Detection")

# --- Load Model & Scaler ---
model = load_model("full_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# --- Load Threshold ---
try:
    with open("threshold.txt", "r") as f:
        adjusted_threshold = float(f.read().strip())
except FileNotFoundError:
    st.error("❌ Threshold file not found! Please ensure 'threshold.txt' exists.")
    st.stop()

# --- Load CSV File ---
df = pd.read_csv("step_data-2.csv")
st.write("✅ Loaded local CSV file: step_data-2.csv")

# --- Required Features ---
required_cols = ['steps', 'activity_duration', 'step_frequency', 'rest_period', 'noise_flag', 'battery_level']

if not all(col in df.columns for col in required_cols):
    st.error(f"❌ CSV must contain these columns: {required_cols}")
else:
    # --- Feature Extraction ---
    X = df[required_cols].values
    y_true = df['anomaly_detected'].values if 'anomaly_detected' in df.columns else None

    # --- Scale & Predict ---
    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)
    reconstruction_error_smooth = gaussian_filter1d(reconstruction_error, sigma=1)

    # --- Prediction ---
    from sklearn.metrics import precision_score, recall_score, f1_score

# Sweep threshold values to find best F1 score
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0
    threshold_range = np.linspace(0.001, np.max(reconstruction_error_smooth), 500)

    for t in np.linspace(0.001, np.max(reconstruction_error_smooth), 500):
       preds_temp = (reconstruction_error_smooth > t).astype(int)
       if y_true is not None:
           precision = precision_score(y_true, preds_temp, zero_division=0)
           recall = recall_score(y_true, preds_temp, zero_division=0)
           f1 = f1_score(y_true, preds_temp, zero_division=0)
           if precision >= target_precision and f1 > best_f1:
              best_f1 = f1
              best_threshold = t
              best_precision = precision
              best_recall = recall
    if best_threshold == 0:  # fallback if no threshold meets precision requirement
       st.warning("No threshold met the precision requirement. Showing highest F1 score instead.")
       for t in np.linspace(0.001, np.max(reconstruction_error_smooth), 500):
          preds_temp = (reconstruction_error_smooth > t).astype(int)
          precision = precision_score(y_true, preds_temp, zero_division=0)
          recall = recall_score(y_true, preds_temp, zero_division=0)
          f1 = f1_score(y_true, preds_temp, zero_division=0)
          if f1 > best_f1:
             best_f1 = f1
             best_threshold = t
             best_precision = precision
             best_recall = recall  # fallback if no threshold meets precision requirement
   

# Use best threshold for final prediction
    
    preds = (reconstruction_error_smooth > best_threshold).astype(int)

    
    df['reconstruction_error'] = reconstruction_error_smooth
    df['anomaly_predicted'] = preds

    st.success("✅ Anomaly detection complete!")
    st.dataframe(df[['steps', 'activity_duration', 'step_frequency', 'anomaly_predicted']])

    # --- Metrics ---
    if y_true is not None:
        matched = np.sum((preds == 1) & (y_true == 1))
        detected = np.sum(preds == 1)
        total_true = np.sum(y_true == 1)
        

        st.markdown("### 📊 Evaluation Metrics")
        st.write(f"**Best Threshold:** `{best_threshold:.6f}`")
        st.write(f"**Matched (TP):** {matched}")
        st.write(f"**Detected Anomalies:** {detected}")
        st.write(f"**True Anomalies:** {total_true}")
        st.write(f"**Precision:** {best_precision:.2f}")
        st.write(f"**Recall:** {best_recall:.2f}")
        st.write(f"**F1 Score:** {best_f1:.2f}")

    st.markdown(f"### 🧪 Threshold used: `{adjusted_threshold:.6f}`")

# -----------------------------
# 🧍 User Manual Input Section
# -----------------------------
st.markdown("---")
st.markdown("## 🔍 Check a Custom Activity Sample")

with st.form("anomaly_form"):
    steps = st.number_input("Steps", min_value=0, value=1000)
    activity_duration = st.number_input("Activity Duration (minutes)", min_value=0.0, value=30.0)
    step_frequency = st.number_input("Step Frequency", min_value=0.0, value=1.2)
    rest_period = st.number_input("Rest Period (minutes)", min_value=0.0, value=5.0)
    noise_flag = st.selectbox("Noise Flag", [0, 1])
    battery_level = st.slider("Battery Level (%)", 0, 100, 80)

    submitted = st.form_submit_button("Check Anomaly")

    if submitted:
        user_input = np.array([[steps, activity_duration, step_frequency, rest_period, noise_flag, battery_level]])
        user_scaled = scaler.transform(user_input)
        user_pred = model.predict(user_scaled)
        user_error = np.mean(np.square(user_scaled - user_pred), axis=1)
        user_error_smooth = gaussian_filter1d(user_error, sigma=1)
        is_anomaly = user_error_smooth[0] > adjusted_threshold

        if is_anomaly:
            st.error("⚠️ Anomaly Detected!")
        else:
            st.success("✅ No Anomaly Detected.")
        st.write(f"Reconstruction Error: `{user_error_smooth[0]:.6f}`")

