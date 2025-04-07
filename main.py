import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Pet Activity Anomaly Detection")

# Load model
model = load_model("model.weights.h5")

st.title("🐾 Pet Activity Anomaly Detection")

# Load CSV directly
df = pd.read_csv("step_data-2.csv")  # replace with your actual file
st.write("✅ Loaded local CSV file: step_data-2.csv")

# Check required columns
required_cols = ['steps', 'activity_duration', 'step_frequency', 'rest_period', 'noise_flag', 'battery_level']
if not all(col in df.columns for col in required_cols):
    st.error(f"CSV must contain these columns: {required_cols}")
else:
    # Prepare data
    X = df[required_cols].values
    y_true = df['anomaly_detected'].values if 'anomaly_detected' in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model prediction
    X_pred = model.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)
    reconstruction_error_smooth = gaussian_filter1d(reconstruction_error, sigma=1)

    # Threshold slider
    threshold = st.slider("Set anomaly threshold", 0.0, float(np.max(reconstruction_error_smooth)), 0.02)
    preds = (reconstruction_error_smooth > threshold).astype(int)

    # Display results
    df['reconstruction_error'] = reconstruction_error_smooth
    df['anomaly_predicted'] = preds

    st.success("✅ Anomaly detection complete!")
    st.dataframe(df[['steps', 'activity_duration', 'step_frequency', 'anomaly_predicted']])

    if y_true is not None:
        matched = np.sum((preds == 1) & (y_true == 1))
        detected = np.sum(preds == 1)
        total_true = np.sum(y_true == 1)
        precision = matched / detected if detected > 0 else 0
        recall = matched / total_true if total_true > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        st.markdown("### 📊 Metrics")
        st.write(f"**Matched:** {matched}")
        st.write(f"**Detected Anomalies:** {detected}")
        st.write(f"**True Anomalies:** {total_true}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")
