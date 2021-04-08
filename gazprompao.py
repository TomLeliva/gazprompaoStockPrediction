# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

plt.style.use('fivethirtyeight')

tickers = ['GAZP.ME', 'SMPC.BK', 'AW9U.SI']

for ticker in tickers:
  df = web.DataReader(ticker, data_source='yahoo', start='1999-01-02', end=datetime.date.today())

  df

  df.shape

  #plt.figure(figsize=(16,8))
  #plt.title('Close Price History')
  #plt.plot(df['Close'])
  #plt.xlabel('Date', fontsize=18)
  #plt.ylabel('Close price', fontsize=18)
  #plt.show()

  data = df.filter(['Close'])
  #convert to numpy array
  dataset = data.values

  training_data_len = math.ceil(len(dataset)*.8)
  training_data_len

  #Scale the data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  scaled_data

  #Create training data set
  train_data = scaled_data[0:training_data_len, :]
  #Split the data into x and y dataset
  x_train = []
  y_train = []

  for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
    if i<=61:
      #print(x_train)
      print(y_train)
      print()

  x_train, y_train = np.array(x_train), np.array(y_train)
  #Reshape the data
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  x_train.shape

  #Build LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  #compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')
  #train the model
  model.fit(x_train, y_train, batch_size=1, epochs=1)

  #Create the testing dataset
  test_DATA = scaled_data[training_data_len-60:,:]
  #Create the data sets x_test and y_test
  x_test = []
  y_test = dataset[training_data_len:, :]
  for i in range(60, len(test_DATA)):
    x_test.append(test_DATA[i-60:i,0])

  #Convert the data to a numpy array
  x_test = np.array(x_test)

  #Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  #Get the model predicted price values
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  #Get the root mean squarred error(RMSE)
  rmse = np.sqrt(np.mean(predictions - y_test)**2)
  rmse

  #Plot the data
  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = predictions
  #Visuallize
  plt.figure(figsize=(16,8))
  plt.title("Model "+ ticker)
  plt.xlabel("Date", fontsize=18)
  plt.ylabel('Close Price', fontsize=18)
  plt.plot(train['Close'])
  plt.plot(valid[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()

  quote = web.DataReader(ticker, data_source='yahoo', start='2011-12-01', end=datetime.date.today())
  #Create a new dataframe
  new_df = quote.filter(['Close'])
  #Get the last 60 day closing price values and convert the dataframe to an array
  last_60_days = new_df[-60:].values
  #Scale the data
  last_60_days_scaled = scaler.transform(last_60_days)
  #Create a list
  X_test = []
  #Append the past 60 days
  X_test.append(last_60_days_scaled)
  # Convert the X_test to numpy array
  X_test = np.array(X_test)
  #Reshape data
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  #get the predicted scaled price
  pred_price = model.predict(X_test)
  #undo the scaling
  pred_price = scaler.inverse_transform(pred_price)
  print(pred_price)