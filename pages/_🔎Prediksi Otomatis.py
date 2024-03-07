import streamlit as st
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from datetime import timedelta

# Streamlit app
def main():
    st.title("Prediksi Crypto Metaverse")
    st.write("Sistem Prediksi menggunakan Algoritma Long Short Term Memory dan Gated Recurrent Unit, sistem ini sudah melakukan pelatihan model sehingga user bisa melakukan prediksi dengan cepat, model dilatih dengan hasil dan parameter yang terbaik.")
    # Sidebar Input Data
    #--------------------#
    st.sidebar.header("Data Download")
    stock_symbol =st.sidebar.selectbox("Masukkan Nama Coin:", ["STX4847-USD", "ICP-USD", "RNDR-USD", "AXS-USD", "WEMIX-USD", "SAND-USD", "THETA-USD", "MANA-USD", "APE-USD", "ENJ-USD", "ZIL-USD", "ILV-USD"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
    # Download stock price data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    #--------------------#

    # Proses Seluruh Data | 4 Future
    #------------------------------#
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
    #------------------------------#


    # Data preparation
    #-----------------#
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
    #-----------------#
    
    # Reshape data for LSTM and GRU models
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


    # Sidebar for model selection
    #---------------------------#
    st.sidebar.header("Select Model")
    model_type = st.sidebar.selectbox("Select Model Type:", ["LSTM", "GRU"])

    # Mengambil Mode
    if model_type == "LSTM":
        final_model = load_model("stx_model_lstm.h5")
    else:
        final_model = load_model("stx_model_gru.h5")
        
    if model_type == "LSTM":
        final_model_high = load_model("high_model_lstm.h5")
    else:
        final_model_high = load_model("high_model_gru.h5")
    
    if model_type == "LSTM":
        final_model_open = load_model("open_model_lstm.h5")
    else:
        final_model_open = load_model("open_model_gru.h5")
        
    if model_type == "LSTM":
        final_model_low = load_model("open_model_lstm.h5")
    else:
        final_model_low = load_model("open_model_gru.h5")
    #---------------------------#

    # Model evaluation
    #-----------------#
    y_pred = final_model.predict(X_test)
    y_pred = scaler_close.inverse_transform(y_pred)
    y_test_orig = scaler_close.inverse_transform(y_test.reshape(-1, 1))
    
    b_pred = final_model_open.predict(A_test)
    b_pred = scaler_open.inverse_transform(b_pred)
    b_test_orig = scaler_open.inverse_transform(b_test.reshape(-1, 1))
    
    d_pred = final_model_high.predict(C_test)
    d_pred = scaler_high.inverse_transform(d_pred)
    d_test_orig = scaler_high.inverse_transform(d_test.reshape(-1, 1))
    
    q_pred = final_model_low.predict(P_test)
    q_pred = scaler_low.inverse_transform(q_pred)
    q_test_orig = scaler_low.inverse_transform(q_test.reshape(-1, 1))

    # Perhitungan Evaluasi
    #--------------------#
    mse_close = mean_squared_error(y_pred, y_test_orig)    #  perhitungan MSE
    rmse_close = math.sqrt(mse_close)                      #  perhitungan RMSE    
    mad_close = np.mean(np.abs(y_test_orig - y_pred))      #  perhitungan MAD
    mape_close = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100
    
    mse_open = mean_squared_error(b_test_orig, b_pred)    #  perhitungan MSE
    rmse_open = math.sqrt(mse_open)                       #  perhitungan RMSE    
    mad_open = np.mean(np.abs(b_test_orig - b_pred))      #  perhitungan MAD
    mape_open = np.mean(np.abs((b_test_orig - b_pred) / b_test_orig)) * 100
    
    mse_high = mean_squared_error(d_test_orig, d_pred)    #  perhitungan MSE
    rmse_high = math.sqrt(mse_high)                       #  perhitungan RMSE    
    mad_high = np.mean(np.abs(d_test_orig - d_pred))      #  perhitungan MAD
    mape_high = np.mean(np.abs((d_test_orig - d_pred) / d_test_orig)) * 100
    
    mse_low = mean_squared_error(q_test_orig, q_pred)    #  perhitungan MSE
    rmse_low = math.sqrt(mse_low)                       #  perhitungan RMSE    
    mad_low = np.mean(np.abs(q_test_orig - q_pred))      #  perhitungan MAD
    mape_low = np.mean(np.abs((q_test_orig - q_pred) / q_test_orig)) * 100
    #-----------------------------------------------#
    
    #-----------------Selected Price----------------#
    CloseTab, OpenTab, HighTab, LowTab, ActualTab, ResultTab = st.tabs(["Close", "Open", "High", "Low", "Actual", "All Result"])
    
    with CloseTab:
        # Display results
        st.header(f"Results Close Price for {model_type} Model")
        st.write("Root Mean Squared Error (RMSE):", round(rmse_close, 5))
        st.write("Mean Absolute Deviation (MAD):", round (mad_close, 5))
        st.write("Mean Absolute Percentage Error (MAPE):", round (mape_close, 5)) 
        
        # Visualize predictions
        visualize_predictions_close(data, train_size_close, n_steps, y_test_orig, y_pred)
        
        # Display combined actual and predicted data table with time information
        st.header("Table Close Harga Asli dan Harga Prediksi")
        
        # Add time information to the header
        st.write("Data range:", data.index[train_size_close + n_steps:].min(), "to", data.index[train_size_close + n_steps:].max())
        
        # Calculate the difference between actual and predicted prices
        price_difference_close = y_test_orig.flatten() - y_pred.flatten()
        
        # Calculate the percentage difference
        percentage_difference_close = (price_difference_close / y_test_orig.flatten()) * 100
        
        # Convert predicted prices to strings and cut off decimal places after the 5th digit
        predicted_prices_str_close = [f"{val:.5f}" for val in y_pred.flatten()]
        
        # Combine data, time information, and price difference into one dataframe with column names
        combined_data_close = pd.DataFrame({
            'Tanggal': data.index.date[train_size_close + n_steps:],
            'Actual_Prices': y_test_orig.flatten(),
            'Predicted_Prices': predicted_prices_str_close,
            'Price_Difference': abs(price_difference_close),
            'Percentage_Difference': abs(percentage_difference_close)
        })
    
        # Format the 'Percentage_Difference' column to include the percentage symbol
        combined_data_close['Percentage_Difference'] = combined_data_close['Percentage_Difference'].map("{:.2f}%".format)

        # Display the combined data table
        st.table(combined_data_close)
        
        
    with OpenTab:
        # Display results
        st.header(f"Results Open Price for {model_type} Model")
        st.write("Root Mean Squared Error (RMSE):", round(rmse_open, 5))
        st.write("Mean Absolute Deviation (MAD):", round (mad_open, 5))
        st.write("Mean Absolute Percentage Error (MAPE):", round (mape_open, 5))
        
        # Visualize predictions
        visualize_predictions_open(data, train_size_open, n_steps, b_test_orig, b_pred)
        
        # Display combined actual and predicted data table with time information
        st.header("Table Open Harga Asli dan Harga Prediksi")
        st.write("Data range:", data.index[train_size_open + n_steps:].min(), "to", data.index[train_size_open + n_steps:].max())
        price_difference_open = b_test_orig.flatten() - b_pred.flatten()
        percentage_difference_open = (price_difference_open / b_test_orig.flatten()) * 100
        predicted_prices_str_open = [f"{val:.5f}" for val in b_pred.flatten()]
        combined_data_open = pd.DataFrame({
            'Tanggal': data.index.date[train_size_open + n_steps:],
            'Actual_Prices': b_test_orig.flatten(),
            'Predicted_Prices': predicted_prices_str_open,
            'Price_Difference': abs(price_difference_open),
            'Percentage_Difference': abs(percentage_difference_open)
        })
        combined_data_open['Percentage_Difference'] = combined_data_open['Percentage_Difference'].map("{:.2f}%".format)
        # Display the combined data table
        st.table(combined_data_open)
        
    
    with HighTab:
        # Display results
        st.header(f"Results High Price for {model_type} Model")
        st.write("Root Mean Squared Error (RMSE):", round(rmse_high, 5))
        st.write("Mean Absolute Deviation (MAD):", round (mad_high, 5))
        st.write("Mean Absolute Percentage Error (MAPE):", round (mape_high, 5))
        
        # Visualize predictions
        visualize_predictions_high(data, train_size_high, n_steps, d_test_orig, d_pred)
        
        # Display combined actual and predicted data table with time information
        st.header("Table High Harga Asli dan Harga Prediksi")
        st.write("Data range:", data.index[train_size_high + n_steps:].min(), "to", data.index[train_size_high + n_steps:].max())
        price_difference_high = d_test_orig.flatten() - d_pred.flatten()
        percentage_difference_high = (price_difference_high / d_test_orig.flatten()) * 100
        predicted_prices_str_high = [f"{val:.5f}" for val in d_pred.flatten()]
        combined_data_high = pd.DataFrame({
            'Tanggal': data.index.date[train_size_high + n_steps:],
            'Actual_Prices': d_test_orig.flatten(),
            'Predicted_Prices': predicted_prices_str_high,
            'Price_Difference': abs(price_difference_high),
            'Percentage_Difference': abs(percentage_difference_high)
        })
        combined_data_high['Percentage_Difference'] = combined_data_high['Percentage_Difference'].map("{:.2f}%".format)
        # Display the combined data table
        st.table(combined_data_high)
        

    with LowTab:
        # Display results
        st.header(f"Results Low Price for {model_type} Model")
        st.write("Root Mean Squared Error (RMSE):", round(rmse_low, 5))
        st.write("Mean Absolute Deviation (MAD):", round (mad_low, 5))
        st.write("Mean Absolute Percentage Error (MAPE):", round (mape_low, 5))
        
        # Visualize predictions
        visualize_predictions_low(data, train_size_low, n_steps, q_test_orig, q_pred)
        
        # Display combined actual and predicted data table with time information
        st.header("Table Low Harga Asli dan Harga Prediksi")
        st.write("Data range:", data.index[train_size_low + n_steps:].min(), "to", data.index[train_size_low + n_steps:].max())
        price_difference_low = q_test_orig.flatten() - q_pred.flatten()
        percentage_difference_low = (price_difference_low / q_test_orig.flatten()) * 100
        predicted_prices_str_low = [f"{val:.5f}" for val in q_pred.flatten()]
        combined_data_low = pd.DataFrame({
            'Tanggal': data.index.date[train_size_low + n_steps:],
            'Actual_Prices': q_test_orig.flatten(),
            'Predicted_Prices': predicted_prices_str_low,
            'Price_Difference': abs(price_difference_low),
            'Percentage_Difference': abs(percentage_difference_low)
        })
        combined_data_low['Percentage_Difference'] = combined_data_low['Percentage_Difference'].map("{:.2f}%".format)
        # Display the combined data table
        st.table(combined_data_low)


    with ActualTab:
        # Display results
        st.header(f"Results Actual Price for {model_type} Model")
        
        # Plot all prices
        fig_all_prices = go.Figure()

        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price', line=dict(color='red')))
        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='green')))
        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='yellow')))
        fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='blue')))

        fig_all_prices.update_layout(
            title='Stock Price History',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Stock Price'),
            legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        )

        # Plot subplots for each individual price
        fig_subplots = make_subplots(rows=2, cols=2, subplot_titles=('Opening Price', 'Closing Price', 'Low Price', 'High Price'))

        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price', line=dict(color='red')), row=1, col=1)
        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Closing Price', line=dict(color='green')), row=1, col=2)
        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='yellow')), row=2, col=1)
        fig_subplots.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='blue')), row=2, col=2)

        fig_subplots.update_layout(title='Stock Price Subplots', showlegend=False)

        st.plotly_chart(fig_all_prices)
        st.plotly_chart(fig_subplots)
        
        # Display combined actual and predicted data table with time information
        st.header("Table All Price")
        
        # Add time information to the header
        st.write("Data range:", start_date, "to", end_date)

        # Combine data, time information, and price difference into one dataframe with column names
        combined_data_all_actual = pd.DataFrame({
            'Close_Predict':data['Close'],
            'Open_Predict': data['Open'],
            'High_Predict': data['High'],
            'Low_Predict': data['Low']

        })
    
        # Display the combined data table
        st.table(combined_data_all_actual)
        
    with ResultTab:
        # Display results
        st.header(f"Results All Price Prediction for {model_type} Model")
        
        # Visualize predictions
        visualize_predictions_all_pred(data, train_size_open, train_size_close, train_size_high, train_size_low, n_steps, b_pred, y_pred, d_pred, q_pred)
        
        # Display combined actual and predicted data table with time information
        st.header("Table All Price Prediction")
        
        # Add time information to the header
        st.write("Data range:", data.index[train_size_close + n_steps:].min(), "to", data.index[train_size_close + n_steps:].max())

        # Combine data, time information, and price difference into one dataframe with column names
        combined_data_all = pd.DataFrame({
            'Tanggal': data.index.date[train_size_close + n_steps:],
            'Close_Predict': predicted_prices_str_close,
            'Open_Predict': predicted_prices_str_open,
            'High_Predict': predicted_prices_str_high,
            'Low_Predict': predicted_prices_str_low

        })
    
        # Display the combined data table
        st.table(combined_data_all)
        
#---------------------------------------#
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
#---------------------------------------#

#Visual Grafik_Close
def visualize_predictions_close(data, train_size_close, n_steps, y_test_orig, y_pred):
    fig_close = go.Figure()

    fig_close.add_trace(go.Scatter(x=data.index[:train_size_close + n_steps],  # Menambahkan n_steps untuk data latih
                             y=data['Close'][:train_size_close + n_steps],  # Menambahkan n_steps untuk data latih
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    fig_close.add_trace(go.Scatter(x=data.index[train_size_close + n_steps:],
                             y=y_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig_close.add_trace(go.Scatter(x=data.index[train_size_close + n_steps:],
                             y=y_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig_close.update_layout(title="Close Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')
    
        

    st.plotly_chart(fig_close)


#Visual Grafik_Open
def visualize_predictions_open(data, train_size_open, n_steps, b_test_orig, b_pred):
    fig_open = go.Figure()

    fig_open.add_trace(go.Scatter(x=data.index[:train_size_open + n_steps],  # Menambahkan n_steps untuk data latih
                             y=data['Open'][:train_size_open + n_steps],  # Menambahkan n_steps untuk data latih
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    fig_open.add_trace(go.Scatter(x=data.index[train_size_open + n_steps:],
                             y=b_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig_open.add_trace(go.Scatter(x=data.index[train_size_open + n_steps:],
                             y=b_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig_open.update_layout(title="Open Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')
    
        

    st.plotly_chart(fig_open)
    

#Visual Grafik_High
def visualize_predictions_high(data, train_size_high, n_steps, d_test_orig, d_pred):
    fig_high = go.Figure()

    fig_high.add_trace(go.Scatter(x=data.index[:train_size_high + n_steps],  # Menambahkan n_steps untuk data latih
                             y=data['High'][:train_size_high + n_steps],  # Menambahkan n_steps untuk data latih
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    fig_high.add_trace(go.Scatter(x=data.index[train_size_high + n_steps:],
                             y=d_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig_high.add_trace(go.Scatter(x=data.index[train_size_high + n_steps:],
                             y=d_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig_high.update_layout(title="High Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')
        
    st.plotly_chart(fig_high)
    

#Visual Grafik_Low
def visualize_predictions_low(data, train_size_low, n_steps, q_test_orig, q_pred):
    fig_low = go.Figure()

    fig_low.add_trace(go.Scatter(x=data.index[:train_size_low + n_steps],  # Menambahkan n_steps untuk data latih
                             y=data['Low'][:train_size_low + n_steps],  # Menambahkan n_steps untuk data latih
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    fig_low.add_trace(go.Scatter(x=data.index[train_size_low + n_steps:],
                             y=q_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig_low.add_trace(go.Scatter(x=data.index[train_size_low + n_steps:],
                             y=q_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig_low.update_layout(title="Low Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')
        
    st.plotly_chart(fig_low)

    
#Visual Grafik All Predict
def visualize_predictions_all_pred(data,train_size_open,train_size_close, train_size_high, train_size_low, n_steps, b_pred, y_pred, d_pred, q_pred):
    fig_all_prices_predict = go.Figure()
    
    fig_all_prices_predict.add_trace(go.Scatter(x=data.index[train_size_open + n_steps:],
                            y=b_pred.flatten(),
                            mode='lines',
                            name="Prediksi Open",
                            line=dict(color='red')))
    
    fig_all_prices_predict.add_trace(go.Scatter(x=data.index[train_size_close + n_steps:],
                             y=y_pred.flatten(),
                             mode='lines',
                             name="Prediksi Close",
                             line=dict(color='green')))
    
    fig_all_prices_predict.add_trace(go.Scatter(x=data.index[train_size_high + n_steps:],
                             y=d_pred.flatten(),
                             mode='lines',
                             name="Prediksi High",
                             line=dict(color='blue')))
    
    fig_all_prices_predict.add_trace(go.Scatter(x=data.index[train_size_low + n_steps:],
                             y=q_pred.flatten(),
                             mode='lines',
                             name="Prediksi Low",
                             line=dict(color='yellow')))
    
    fig_all_prices_predict.update_layout(title="All Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Stock Price (USD)",
                      template='plotly_dark')
    
    st.plotly_chart(fig_all_prices_predict)

if __name__ == "__main__":
    main()
