import numpy as np
import pandas as pd
import pandas_datareader as data
from plotly import graph_objs as go
from datetime import date
import streamlit as st
import yfinance as yf
from os import path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

from sklearn.preprocessing import MinMaxScaler


@st.cache_data
def load_data(ticker, start):
    data = yf.download(ticker, start, date.today())
    data.reset_index(inplace=True)
    print(data)
    return data


@st.cache_data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Open'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    return fig


@st.cache_data
def load_model(time_step=15):
    # check if there is a model named "model-[current_date].keras" in the current directory
    # if not, train the model and save it, else load the model
    data = load_data("IDR=X", date.today().replace(
        year=date.today().year - 1))
    df_train = data[['Date', 'Close']]
    df_train.set_index('Date', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(df_train).reshape(-1, 1))

    training_size = int(len(closedf) * 0.60)
    # test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size,
                                    :], closedf[training_size:len(closedf), :1]

    if path.exists(f"model-{date.today()}.keras"):
        model = tf.keras.models.load_model(f"model-{date.today()}.keras")
        return model, test_data, scaler

    # model does not exist, train the model for today's data
    else:
        # Load data from 1 year ago to today
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create the Stacked LSTM model
        model = Sequential()
        model.add(Input(shape=(None, 1)))
        model.add(LSTM(10, activation="relu"))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X_train, y_train, validation_data=(
            X_test, y_test), epochs=80, batch_size=32, verbose=1)

        # Save the model
        model.save(f"model-{date.today()}.keras")

        return model, test_data, scaler


# maximum date is yesterday
TODAY = st.date_input('Enter Start Date',
                      max_value=date.today().replace(day=date.today().day - 1),
                      value=date.today().replace(day=date.today().day - 1))

st.title("Nilai Tukar Mata uang")
stocks = ("IDR=X")


data_load_state = st.text("Load data...")
data = load_data(stocks, TODAY)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.dataframe(data)


st.plotly_chart(plot_raw_data(data))

# Forecasting
time_step = 15
model, test_data, scaler = load_model(time_step)

# Forecasting for the next 30 days
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []
n_steps = time_step
i = 0
pred_days = 30

while (i < pred_days):
    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

lst_output = scaler.inverse_transform(lst_output)
forecast_dates = pd.date_range(
    data['Date'].iloc[-1], periods=pred_days + 1).tolist()

# Create a plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
fig.add_trace(go.Scatter(x=forecast_dates, y=np.concatenate(
    [data['Close'].values[-1:], lst_output.flatten()]), name='Forecast'))

fig.update_layout(title_text='LSTM Forecasting',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  legend_title_text='Series',
                  plot_bgcolor='white',
                  font_size=15,
                  font_color='black')

st.plotly_chart(fig)
st.subheader('Predicted Close Price for Next 30 Days')
data_forecast = pd.DataFrame(
    {'Date': forecast_dates[1:], 'Close': lst_output.flatten()})
st.write(data_forecast)
