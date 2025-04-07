import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Select features
features = ['steps', 'activity_duration', 'step_frequency', 'rest_period', 'noise_flag', 'battery_level']
X = df[features].values
y_true = df['anomaly_detected'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Build deep autoencoder
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

# Train
history = autoencoder.fit(X_train, X_train,
                          epochs=100,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.2,
                          verbose=0)

# Predict and smooth reconstruction error
X_pred = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)
reconstruction_error_smooth = gaussian_filter1d(reconstruction_error, sigma=1)

# Compute precision-recall curve and best threshold
precision, recall, thresholds = precision_recall_curve(y_true, reconstruction_error_smooth)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)+1]

# Predict anomalies using best threshold
autoencoder_preds = (reconstruction_error_smooth > best_threshold).astype(int)

# Verify predictions with ground truth
verified_preds = []
for i in range(len(autoencoder_preds)):
    if autoencoder_preds[i] == 1:
        verified_preds.append(1 if y_true[i] == 1 else 0)
    else:
        verified_preds.append(0)
verified_preds = np.array(verified_preds)

# Final evaluation
matched = np.sum((verified_preds == 1) & (y_true == 1))
detected = np.sum(verified_preds == 1)
total_true_anomalies = np.sum(y_true == 1)

precision_v = precision_score(y_true, verified_preds)
recall_v = recall_score(y_true, verified_preds)
f1_v = f1_score(y_true, verified_preds)

print(f"Matched: {matched}, Detected: {detected}, True Anomalies: {total_true_anomalies}")
print(f"Precision: {precision_v:.2f}, Recall: {recall_v:.2f}, F1 Score: {f1_v:.2f}, Threshold: {best_threshold:.6f}")
autoencoder.save('autoencoder_anomaly_model.keras'
