import streamlit as st
import appdirs as ad
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Streamlit app
def main():
    st.title("Prediksi Crypto Metaverse")
    st.write("Sistem Prediksi menggunakan Algoritma Long Short Term Memory dan Gated Recurrent Unit, sistem ini sudah melakukan pelatihan model sehingga user bisa melakukan prediksi dengan cepat, model dilatih dengan hasil dan parameter yang terbaik.")
    
    # Sidebar Input Data
    st.sidebar.header("Data Download")
    stock_symbol = st.sidebar.selectbox("Masukkan Nama Coin:", ["STX4847-USD", "ICP-USD", "RNDR-USD", "AXS-USD", "WEMIX-USD", "SAND-USD", "THETA-USD", "MANA-USD", "APE-USD", "ENJ-USD", "ZIL-USD", "ILV-USD", "SUSHI-USD", "MIC27033-USD", "HIGH-USD", "FLOKI-USD", "EGLD-USD", "MASK8536-USD"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
    # Download stock price data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Proses Seluruh Data | 4 Future
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaled_data_close = scaler_close.fit_transform(close_prices)
    
    open_prices = data['Open'].values.reshape(-1, 1)
    scaler_open = MinMaxScaler(feature_range=(0, 1))
    scaled_data_open = scaler_open.fit_transform(open_prices)
    
    high_prices = data['High'].values.reshape(-1, 1)
    scaler_high = MinMaxScaler(feature_range=(0, 1))
    scaled_data_high = scaler_high.fit_transform(high_prices)
    
    low_prices = data['Low'].values.reshape(-1, 1)
    scaler_low = MinMaxScaler(feature_range=(0, 1))
    scaled_data_low = scaler_low.fit_transform(low_prices)

    # Data preparation
    n_steps = 120
    X, y = prepare_data_close(scaled_data_close, n_steps)
    A, b = prepare_data_open(scaled_data_open, n_steps)
    C, d = prepare_data_high(scaled_data_high, n_steps)
    P, q = prepare_data_low(scaled_data_low, n_steps)

    # Splitting into train and test sets
    train_size_close = int(len(X) * 0.8)
    X_train, X_test = X[:train_size_close], X[train_size_close:]
    y_train, y_test = y[:train_size_close], y[train_size_close:]
    
    train_size_open = int(len(A) * 0.8)
    A_train, A_test = A[:train_size_open], A[train_size_open:]
    b_train, b_test = b[:train_size_open], b[train_size_open:]
    
    train_size_high = int(len(C) * 0.8)
    C_train, C_test = C[:train_size_high], C[train_size_high:]
    d_train, d_test = d[:train_size_high], b[train_size_high:]
    
    train_size_low = int(len(P) * 0.8)
    P_train, P_test = P[:train_size_low], P[train_size_low:]
    q_train, q_test = q[:train_size_low], q[train_size_low:]

    # Reshape data for LSTM and GRU models
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Load both models
    final_model_lstm = load_model("stx_model_lstm.h5")
    final_model_gru = load_model("stx_model_gru.h5")
    

    # Model evaluation for LSTM
    y_pred_lstm = final_model_lstm.predict(X_test_lstm)
    y_pred_lstm = scaler_close.inverse_transform(y_pred_lstm)
    
    
    # Model evaluation for GRU
    y_pred_gru = final_model_gru.predict(X_test_gru)
    y_pred_gru = scaler_close.inverse_transform(y_pred_gru)
    
    # Denormalize the actual stock prices
    y_test_denorm = scaler_close.inverse_transform(y_test.reshape(-1, 1))
    
    st.header(f"Results Close Price {stock_symbol} for LSTM and GRU Models")
    st.write("LSTM - Root Mean Squared Error (RMSE):", round(math.sqrt(mean_squared_error(y_pred_lstm, y_test_denorm)), 5))
    st.write("LSTM - Mean Absolute Error (MAE):", round(np.mean(np.abs(y_pred_lstm - y_test_denorm)), 5))
    st.write("LSTM - Mean Absolute Percentage Error (MAPE):", round(np.mean(np.abs((y_pred_lstm - y_test_denorm) / y_test_denorm)) * 100, 5))
        
    st.write("GRU - Root Mean Squared Error (RMSE):", round(math.sqrt(mean_squared_error(y_pred_gru, y_test_denorm)), 5))
    st.write("GRU - Mean Absolute Error (MAE):", round(np.mean(np.abs(y_pred_gru - y_test_denorm)), 5))
    st.write("GRU - Mean Absolute Percentage Error (MAPE):", round(np.mean(np.abs((y_pred_gru - y_test_denorm) / y_test_denorm)) * 100, 5))
        
    visualize_combined_predictions(data, train_size_close, n_steps, y_test_denorm, y_pred_lstm, y_pred_gru, "Close")

def prepare_data_close(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        lag_values_close = data[i:(i + n_steps), 0]
        X.append(np.concatenate([lag_values_close, [data[i + n_steps, 0]]]))
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def prepare_data_open(data, n_steps):
    A, b = [], []
    for i in range(len(data) - n_steps):
        lag_values_open = data[i:(i + n_steps), 0]
        A.append(np.concatenate([lag_values_open, [data[i + n_steps, 0]]]))
        b.append(data[i + n_steps, 0])
    return np.array(A), np.array(b)

def prepare_data_high(data, n_steps):
    C, d = [], []
    for i in range(len(data) - n_steps):
        lag_values_high = data[i:(i + n_steps), 0]
        C.append(np.concatenate([lag_values_high, [data[i + n_steps, 0]]]))
        d.append(data[i + n_steps, 0])
    return np.array(C), np.array(d)

def prepare_data_low(data, n_steps):
    P, q = [], []
    for i in range(len(data) - n_steps):
        lag_values_low = data[i:(i + n_steps), 0]
        P.append(np.concatenate([lag_values_low, [data[i + n_steps, 0]]]))
        q.append(data[i + n_steps, 0])
    return np.array(P), np.array(q)

#Visual Combined Grafik
def visualize_combined_predictions(data, train_size, n_steps, y_test, y_pred_lstm, y_pred_gru, price_type):
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=data.index[:train_size + n_steps],
                             y=data[price_type][:train_size + n_steps],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    # Adjust y_test to align correctly with the indices
    actual_indices = data.index[train_size + n_steps: train_size + n_steps + len(y_test)]

    # Add actual stock prices
    fig.add_trace(go.Scatter(x=actual_indices,
                             y=y_test.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    # Add LSTM predictions
    fig.add_trace(go.Scatter(x=actual_indices,
                             y=y_pred_lstm.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices (LSTM)",
                             line=dict(color='red')))

    # Add GRU predictions
    fig.add_trace(go.Scatter(x=actual_indices,
                             y=y_pred_gru.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices (GRU)",
                             line=dict(color='green')))

    fig.update_layout(title=f"{price_type} Price Prediction for LSTM & GRU",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
