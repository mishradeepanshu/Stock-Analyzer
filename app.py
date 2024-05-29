import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt



start = '2012-01-01'
end = '2023-12-31'


st.title('Stock Market Predictor')

user_input = st.text_input('Enter Stock Symbol', 'AAPL')
df = yf.download(user_input, start ,end)

#Describe the df
st.subheader('Data From 2012-2023')
st.write(df.describe())


#visulaize the closing price
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100= df['Close'].rolling(window=100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, label='100MA')
plt.title('Closing Price vs Time Chart with 100MA')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='upper left')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma100= df['Close'].rolling(window=100).mean()
ma200= df['Close'].rolling(window=200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma200, label='200MA')
plt.plot(ma100, label='100MA')
plt.title('Closing Price vs Time Chart with 100MA and 200MA')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='upper left')
st.pyplot(fig)


#Split the df into 70% training and 30% testing
data_training = pd.DataFrame(df.Close[0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df.Close[int(len(df)*0.70): len(df)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#load model
model = load_model('stock_prediction.h5')

#test the model
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#Visualize the data
st.subheader('Predictions vs Real Data')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, label='Real Data')
plt.plot(y_predicted, label='Predicted Data')
plt.title('Predictions vs Real Data')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='upper left')
st.pyplot(fig2)


