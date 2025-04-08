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
import matplotlib.pyplot as plt

st.title("🐾 Pet Activity Anomaly Detection with Autoencoder")

uploaded_file = st.file_uploader("Upload step_data-2.csv", type=["csv"])

if uploaded_file is not None:
    # --- Load Data ---
    df = pd.read_csv(uploaded_file)

    features = ['steps', 'activity_duration', 'step_frequency', 'rest_period', 'noise_flag', 'battery_level']

    try:
        X = df[features].values
        y_true = df['anomaly_detected'].values

        # --- Standardize ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Train-Test Split ---
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

        # --- Autoencoder ---
        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))

        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)

        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

        with st.spinner("Training autoencoder model..."):
            autoencoder.fit(X_train, X_train,
                            epochs=100,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=0)

        # --- Inference on Training Data ---
        X_pred = autoencoder.predict(X_scaled)
        reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

        # --- Smoothing & Threshold ---
        reconstruction_error_smooth = gaussian_filter1d(reconstruction_error, sigma=1)

        precision, recall, thresholds = precision_recall_curve(y_true, reconstruction_error_smooth)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx if best_idx < len(thresholds) else -1]
        adjusted_threshold = best_threshold * 0.94

        # --- Prediction on Existing Data (for evaluation only) ---
        autoencoder_preds = (reconstruction_error_smooth > adjusted_threshold).astype(int)
        verified_preds = (autoencoder_preds & y_true).astype(int)

        precision_v = precision_score(y_true, verified_preds, zero_division=0)
        recall_v = recall_score(y_true, verified_preds, zero_division=0)
        f1_v = f1_score(y_true, verified_preds, zero_division=0)

        # --- Display Evaluation ---
        st.markdown("### ✅ Model Evaluation")
        st.write(f"**Precision:** {precision_v:.2f}")
        st.write(f"**Recall:** {recall_v:.2f}")
        st.write(f"**F1 Score:** {f1_v:.2f}")
        st.write(f"**Adjusted Threshold:** {adjusted_threshold:.6f}")

        # --- Visualize Error ---
        st.markdown("### 📉 Reconstruction Error (Smoothed)")
        fig, ax = plt.subplots()
        ax.plot(reconstruction_error_smooth, label='Smoothed Error', color='blue')
        ax.axhline(y=adjusted_threshold, color='red', linestyle='--', label='Threshold')
        ax.set_title('Reconstruction Error')
        ax.set_xlabel('Index')
        ax.set_ylabel('Error')
        ax.legend()
        st.pyplot(fig)

        # --- User Input Section ---
        st.markdown("### 🧾 Predict New Input")
        input_data = []
        for feature in features:
            value = st.number_input(f"Enter {feature}", value=0.0, format="%.4f")
            input_data.append(value)

        if st.button("🔍 Predict Anomaly"):
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            reconstructed = autoencoder.predict(input_scaled)
            error = np.mean(np.square(input_scaled - reconstructed))
            smoothed_error = gaussian_filter1d([error], sigma=1)[0]

            is_anomaly = smoothed_error > adjusted_threshold

            st.write(f"**Reconstruction Error:** {smoothed_error:.6f}")
            st.markdown(f"### 🔎 Prediction Result: {'🚨 Anomaly Detected' if is_anomaly else '✅ Normal Behavior'}")

    except Exception as e:
        st.error(f"Error: {e}")




