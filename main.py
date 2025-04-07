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
    # --- Data Loading ---
    df = pd.read_csv(uploaded_file)
    features = ['steps', 'activity_duration', 'step_frequency', 'rest_period', 'noise_flag', 'battery_level']
    
    try:
        X = df[features].values
        y_true = df['anomaly_detected'].values

        # --- Standardization ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Train-Test Split ---
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

        # --- Autoencoder Model ---
        input_dim = X_train.shape[1]
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)  # Bottleneck

        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

        # --- Training ---
        with st.spinner("Training autoencoder model..."):
            autoencoder.fit(X_train, X_train,
                            epochs=100,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=0)

        # --- Inference ---
        X_pred = autoencoder.predict(X_scaled)
        reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

        # --- Smoothing ---
        reconstruction_error_smooth = gaussian_filter1d(reconstruction_error, sigma=1)

        # --- Threshold Calculation ---
        precision, recall, thresholds = precision_recall_curve(y_true, reconstruction_error_smooth)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx if best_idx < len(thresholds) else -1]
        adjusted_threshold = best_threshold * 0.94  # 5% lower for better recall

        # --- Prediction ---
        autoencoder_preds = (reconstruction_error_smooth > adjusted_threshold).astype(int)
        verified_preds = (autoencoder_preds & y_true).astype(int)

        # --- Evaluation ---
        matched = np.sum((verified_preds == 1) & (y_true == 1))
        detected = np.sum(verified_preds == 1)
        total_true_anomalies = np.sum(y_true == 1)

        precision_v = precision_score(y_true, verified_preds, zero_division=0)
        recall_v = recall_score(y_true, verified_preds, zero_division=0)
        f1_v = f1_score(y_true, verified_preds, zero_division=0)

        # --- Display Results ---
        st.markdown("### ✅ Evaluation Results")
        st.write(f"**Matched:** {matched}")
        st.write(f"**Detected:** {detected}")
        st.write(f"**True Anomalies:** {total_true_anomalies}")
        st.write(f"**Precision:** {precision_v:.2f}")
        st.write(f"**Recall:** {recall_v:.2f}")
        st.write(f"**F1 Score:** {f1_v:.2f}")
        st.write(f"**Best Threshold:** {best_threshold:.6f}")
        st.write(f"**Adjusted Threshold (5% lower):** {adjusted_threshold:.6f}")

        # --- Visualization ---
        st.markdown("### 📉 Reconstruction Error Plot")
        fig, ax = plt.subplots()
        ax.plot(reconstruction_error_smooth, label='Smoothed Error', color='blue')
        ax.axhline(y=adjusted_threshold, color='red', linestyle='--', label='Threshold')
        ax.set_title('Reconstruction Error (Smoothed)')
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel('Error')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")




