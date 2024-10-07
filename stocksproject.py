#Libraries
import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np 

from tensorflow import keras 

from datetime import datetime 

#Dataset selection
valid_choice = False
while not valid_choice:
    print('Symbols:\nSPY\nMETA\nAMZN\nAAPL\nNFLX\nGOOG\n')
    print('Please type a symbol from the above options to select ')
    choice = input('the stock whose price you would like to predict: ').upper()
    
    if choice in ['SPY', 'META', 'AMZN', 'AAPL', 'NFLX', 'GOOG']:
        try:
            data = pd.read_csv(f'{choice}.csv')
            title = f'{choice} Close Price Over Time'
            valid_choice = True
        except FileNotFoundError:
            print(f"Error: CSV file for {choice} not found. Please try again.")
    else:
        print('Invalid symbol. Please choose from the provided options.')

# Continue with the rest of the code only if a valid choice was made
if valid_choice:
    num_epochs = int(input('Please type the number of epochs: '))
    print(data.shape) 
    print(data.sample(7)) 
    data.info()
    
    # ... (rest of the code remains unchanged)
else:
    print("No valid stock symbol was selected. Exiting the program.")

#Visualize dataset
data['Date'] = pd.to_datetime(data['Date'])
plt.plot(figsize=(15, 8)) 

prediction_range = data.loc[(data['Date'] > datetime(2023,1,1)) 
 & (data['Date']<datetime(2024,1,1))]
time = data['Date']
close = data['Close']
volume = data['Volume']

fig, ax1 = plt.subplots()

plt.xlabel('Date') 
plt.ylabel('Close Price in $')
plt.plot(time,close)
plt.grid()
#plt.tick_params(axis=close)

ax2 = ax1.twinx()
plt.ylabel('Volume in $100 Millions')
plt.plot(time,volume,color='orange')
plt.legend(labels=['Volume', 'Close'],loc=2)  
#plt.tick_params(axis=volume)

plt.title(title) 
fig.tight_layout()
plt.show()

#Select subset for training data
close_data = data.filter(['Close'])
training_dataset = close_data.values
training = int(np.ceil(len(training_dataset) * .90))
print(training)

#Data preperation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(training_dataset)

training_data = scaled_data
x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Gated Recurrent Neural Network - LSTM

'''
"lstm_model" refers to a vanilla long-short term memory model with a single hidden layer 
of LSTM units, allowing this program to predict the behavior of the closing 
price based soley on historial values of the closing price. In other words, the vanilla 
LSTM model could be applied to any time-series data, not just closing price.
'''
lstm_model = keras.models.Sequential()
lstm_model.add(keras.layers.LSTM(units=64, return_sequences = True, 
                            input_shape = (x_train.shape[1], 1))) 
lstm_model.add(keras.layers.LSTM(units=64)) 
lstm_model.add(keras.layers.Dense(32)) 
lstm_model.add(keras.layers.Dropout(0.5)) 
lstm_model.add(keras.layers.Dense(1)) 
lstm_model.summary 
    
#lstm_model training
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error') 

lstm_model.fit(x_train, y_train, epochs=num_epochs) 
#Testing Data

test_data = scaled_data[training - 60:, :] 
x_test = [] 
y_test = training_dataset[training:, :] 

for i in range(60, len(test_data)): 

    x_test.append(test_data[i-60:i, 0]) 

x_test = np.array(x_test) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 

  
# predict the testing data 

predictions = lstm_model.predict(x_test) 

predictions = scaler.inverse_transform(predictions) 

  
# evaluation metrics 

mse = np.mean(((predictions - y_test) ** 2)) 
print("MSE", mse) 
print("RMSE", np.sqrt(mse)) 

#Visualization
train = data[:training]
test = data[training:]
test['Predictions'] = predictions
plt.figure(figsize=(10, 8)) 
plt.plot(train['Date'], train['Close']) 
plt.plot(test['Date'], test[['Close', 'Predictions']]) 
plt.title(title)
plt.xlabel('Date') 
plt.ylabel("Close") 

plt.legend(['Train', 'Test', 'Predictions'])
plt.grid()
plt.show() 
