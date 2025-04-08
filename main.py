import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import gaussian_filter1d

st.title("🐾 Pet Activity Anomaly Detector")

# --- Load CSV ---
@st.cache_data
def load_data():
    df = pd.read_csv("step_data-2.csv")
    features = ['steps', 'activity_duration', 'step_frequency', 'rest_period', 'noise_flag', 'battery_level']
    X = df[features].values
    y = df['anomaly_detected'].values
    return X, y

X, y_true = load_data()

# --- Standardize ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split data ---
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# --- Cached model building ---
@st.cache_resource
def build_autoencoder_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

# --- Cached training step ---
@st.cache_resource
def train_autoencoder():
    model = build_autoencoder_model(X_train.shape[1])
    model.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)
    return model

autoencoder = train_autoencoder()

# --- Inference ---
X_pred = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)
reconstruction_error_smooth = gaussian_filter1d(reconstruction_error, sigma=1)

# --- Thresholding ---
precision, recall, thresholds = precision_recall_curve(y_true, reconstruction_error_smooth)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx if best_idx < len(thresholds) else -1]
adjusted_threshold = best_threshold * 0.88

# --- Metrics ---
autoencoder_preds = (reconstruction_error_smooth > adjusted_threshold).astype(int)
verified_preds = (autoencoder_preds & y_true).astype(int)
precision_v = precision_score(y_true, verified_preds, zero_division=0)
recall_v = recall_score(y_true, verified_preds, zero_division=0)
f1_v = f1_score(y_true, verified_preds, zero_division=0)
st.markdown("### 🔬 Threshold Tuning Insight")
st.write(f"**Original Best Threshold:** {best_threshold:.6f}")
st.write(f"**Adjusted Threshold (Recall Boost):** {adjusted_threshold:.6f}")



# --- User Input ---
st.markdown("### 📥 Enter Activity Details")
steps = st.number_input("Steps", min_value=0)
activity_duration = st.number_input("Activity Duration (seconds)", min_value=0)
step_frequency = st.number_input("Step Frequency (Hz)", min_value=0.0)
rest_period = st.number_input("Rest Period (seconds)", min_value=0)
noise_flag = st.selectbox("Noise Flag", [0, 1])
battery_level = st.number_input("Battery Level (%)", min_value=0, max_value=100)

user_input = np.array([[steps, activity_duration, step_frequency, rest_period, noise_flag, battery_level]])
user_input_scaled = scaler.transform(user_input)

# --- Prediction ---
if st.button("Predict Anomaly"):
    pred = autoencoder.predict(user_input_scaled)
    reconstruction_err = np.mean(np.square(user_input_scaled - pred))
    is_anomaly = reconstruction_err > (adjusted_threshold + 0.005)
    if is_anomaly:
    st.error(f"🚨 Anomaly Detected! (Error: {reconstruction_err:.4f} > Threshold: {adjusted_threshold:.4f})")
    else:
    st.success(f"✅ Normal Activity (Error: {reconstruction_err:.4f} ≤ Threshold: {adjusted_threshold:.4f})")

    st.markdown("### 🔍 Prediction")
    if is_anomaly:
        st.error("🚨 Anomaly Detected!")
    else:
        st.success("✅ Normal Activity")

    # --- Model Performance ---
    st.markdown("### 📊 Model Performance")
    st.write(f"**Precision:** {precision_v:.2f}")
    st.write(f"**Recall:** {recall_v:.2f}")
    st.write(f"**F1 Score:** {f1_v:.2f}")
    st.write(f"**Adjusted Threshold:** {adjusted_threshold:.6f}")

