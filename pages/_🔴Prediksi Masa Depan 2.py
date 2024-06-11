import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ambil data dari Yahoo Finance
def get_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df['Close']

# Pra-pemrosesan data
def preprocess_data(data, time_step=121):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    x_train, y_train = [], []
    for i in range(time_step, len(data_scaled)):
        x_train.append(data_scaled[i-time_step:i, 0])
        y_train.append(data_scaled[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

# Prediksi masa depan
def predict_future(model, last_sequence, steps_ahead, scaler):
    predictions = []
    last_sequence = np.array(last_sequence)
    
    for _ in range(steps_ahead):
        input_seq = last_sequence.reshape((1, last_sequence.shape[0], 1))
        pred = model.predict(input_seq)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit UI
st.title('Cryptocurrency Price Prediction')
ticker = st.selectbox("Masukkan Nama Coin:", ["STX4847-USD", "ICP-USD", "RNDR-USD", "AXS-USD", "WEMIX-USD", "SAND-USD", "THETA-USD", "MANA-USD", "APE-USD", "ENJ-USD", "ZIL-USD", "ILV-USD", "EGLD-USD", "MASK8536-USD", "SUSHI-USD", "MIC27033-USD"])
start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))
steps_ahead = st.slider('Days to Predict Ahead', 1, 180, 30)

# File paths for saved models
lstm_model_path = 'stx_model_lstm.h5'
gru_model_path = 'stx_model_gru.h5'

if st.button('Predict'):
    data = get_data(ticker, start_date, end_date)

    time_step = 121
    x_train, y_train, scaler = preprocess_data(data, time_step)

    # Load saved models
    lstm_model = load_model(lstm_model_path)
    gru_model = load_model(gru_model_path)

    last_sequence = data[-time_step:].values
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1)).flatten()

    lstm_predictions = predict_future(lstm_model, last_sequence, steps_ahead, scaler)
    gru_predictions = predict_future(gru_model, last_sequence, steps_ahead, scaler)

    future_dates = pd.date_range(start=data.index[-1], periods=steps_ahead + 1, freq='D')[1:]

    # Plot with Plotly
    fig = make_subplots()

    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=future_dates, y=lstm_predictions.flatten(), mode='lines', name='LSTM Predictions'))
    fig.add_trace(go.Scatter(x=future_dates, y=gru_predictions.flatten(), mode='lines', name='GRU Predictions'))

    fig.update_layout(
        title=f'{ticker} Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig)