import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Ambil data dari Yahoo Finance
def get_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df[['Close', 'MA50', 'MA200']].dropna()

# Pra-pemrosesan data
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(time_step, len(data_scaled)):
        x_train.append(data_scaled[i-time_step:i])
        y_train.append(data_scaled[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train, scaler

# Buat dan latih model LSTM
def create_lstm_model(x_train, y_train, epochs=100, batch_size=32):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
    return model

# Buat dan latih model GRU
def create_gru_model(x_train, y_train, epochs=100, batch_size=32):
    model = Sequential()
    model.add(GRU(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
    return model

# Prediksi masa depan
def predict_future(model, last_sequence, steps_ahead, scaler, time_step=60):
    predictions = []
    last_sequence = last_sequence.reshape((1, time_step, last_sequence.shape[1]))
    
    for _ in range(steps_ahead):
        pred = model.predict(last_sequence)
        predictions.append(pred[0, 0])
        # Membuat array dengan dimensi yang sesuai untuk prediksi berikutnya
        pred_full = np.zeros((1, 1, last_sequence.shape[2]))
        pred_full[0, 0, 0] = pred
        last_sequence = np.append(last_sequence[:, 1:, :], pred_full, axis=1)
    
    predictions = np.array(predictions).reshape(-1, 1)
    # Hanya menggunakan scaler untuk kolom 'Close'
    inverse_predictions = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], 2))], axis=1)
    )[:, 0].reshape(-1, 1)
    inverse_predictions = np.maximum(inverse_predictions, 0)
    return inverse_predictions

# Streamlit UI
st.title('Cryptocurrency Price Prediction')
ticker = st.selectbox("Masukkan Nama Coin:", ["STX4847-USD", "ICP-USD", "RNDR-USD", "AXS-USD", "WEMIX-USD", "SAND-USD", "THETA-USD", "MANA-USD", "APE-USD", "ENJ-USD", "ZIL-USD", "ILV-USD", "EGLD-USD", "MASK8536-USD","HIGH-USD", "SUSHI-USD", "MIC27033-USD"])
start_date = st.date_input('Start Date', pd.to_datetime('2022-05-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-05-01'))
steps_ahead = st.slider('Days to Predict Ahead', 1, 180, 30)

if st.button('Predict'):
    data = get_data(ticker, start_date, end_date)
    
    time_step = 60
    x_train, y_train, scaler = preprocess_data(data, time_step)

    lstm_model = create_lstm_model(x_train, y_train)
    gru_model = create_gru_model(x_train, y_train)

    last_sequence = data[['Close', 'MA50', 'MA200']].values[-time_step:]
    last_sequence = scaler.transform(last_sequence)

    lstm_predictions = predict_future(lstm_model, last_sequence, steps_ahead, scaler, time_step)
    gru_predictions = predict_future(gru_model, last_sequence, steps_ahead, scaler, time_step)

    future_dates = pd.date_range(start=data.index[-1], periods=steps_ahead + 1, freq='D')[1:]

    # Plot with Plotly
    fig = make_subplots()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=future_dates, y=lstm_predictions.flatten(), mode='lines', name='LSTM Predictions'))
    fig.add_trace(go.Scatter(x=future_dates, y=gru_predictions.flatten(), mode='lines', name='GRU Predictions'))

    fig.update_layout(
        title=f'{ticker} Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig)

    # Display predictions in a table
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM Prediction': lstm_predictions.flatten().round(5),
        'GRU Prediction': gru_predictions.flatten().round(5)
    })

    st.write("Predicted Prices:")
    st.dataframe(prediction_df)