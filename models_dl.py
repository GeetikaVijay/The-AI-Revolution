import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

def train_lstm(X_train, y_train, epochs=20, batch_size=64):
    """
    Treat feature vectors as sequences of length 1 for simplicity.
    For richer sequence modeling, rebuild X as sliding windows and increase timesteps.
    """
    # Scale Y for better convergence
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).astype("float32")

    # Reshape X to (samples, timesteps, features)
    X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1])).astype("float32")

    model = models.Sequential([
        layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X_train_seq, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler_y

def predict_lstm(model, X_test, scaler_y):
    X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1])).astype("float32")
    y_pred_scaled = model.predict(X_test_seq, verbose=0).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    return y_pred
