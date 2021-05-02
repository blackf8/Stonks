import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt
from pandas import datetime
import math, time
import itertools
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable


stocks = ['VZ', 'T', 'WMT', 'MGM', 'GPS', 'GT', 'BBY', 'AFG', 'ERJ', 'MYE', 'ECPG', 'GCO', 'MPC', 'TRI', 'UFI']

for stock in stocks:

    df = pd.read_csv("data/{}.csv".format(stock), parse_dates=True, index_col=0)
    df = df[['Close']].fillna(method='ffill')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    def load_data(stock, look_back):
        data_raw = stock.values # convert to numpy array
        data = []
        
        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back): 
            data.append(data_raw[index: index + look_back])
        
        data = np.array(data)
        test_set_size = 5
        train_set_size = data.shape[0] - (test_set_size)
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        
        return [x_train, y_train, x_test, y_test]

    look_back = 20 # choose sequence length
    x_train, y_train, x_test, y_test = load_data(df, look_back)
    '''print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)'''

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1

    # Here we define our model as a class
    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.num_layers = num_layers

            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, feature_dim)
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

            # Readout layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # Initialize cell state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            # out.size() --> 100, 32, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
            out = self.fc(out[:, -1, :]) 
            # out.size() --> 100, 10
            return out
        
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


    num_epochs = 100
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim =look_back-1  

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()
        
        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        #if t % 10 == 0:
            #print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()


    '''plt.plot(hist, label="Training loss")
    plt.legend()
    plt.show()'''


    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))


    '''
    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    axes.plot(df[len(df)-len(y_test):].index, y_test, color = 'red', label = 'Real Stock Price')
    axes.plot(df[len(df)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    '''


    prev = y_test[-1].squeeze(0)
    raw_classification_pred = [prev]
    for i in y_test_pred:
        raw_classification_pred.append(i.squeeze(0))
    raw_classification_true = [prev]
    for i in y_test:
        raw_classification_true.append(i.squeeze(0))
    classification_pred = []
    for i in range(len(raw_classification_pred)-1):
        classification_pred.append((raw_classification_pred[i+1]-raw_classification_pred[i])/raw_classification_pred[i]*100)
    classification_true = []
    for i in range(len(raw_classification_true)-1):
        classification_true.append((raw_classification_true[i+1]-raw_classification_true[i])/raw_classification_true[i]*100)


    def calculate_results(stock, classification_pred):
        results = [stock, 0, 0, 0, 0, 0]
        for i in range(5):
            if (classification_pred[i] >= 5):
                results[5] = 1
                results[4] = 1
            elif (classification_pred[i] >= 2 and classification_pred[i] < 5):
                results[4] = 1
            elif (classification_pred[i] <= -2 and classification_pred[i] > -5):
                results[2] = 1
            elif (classification_pred[i] <= -5):
                results[1] = 1
                results[2] = 1
        if 1 not in results:
            results[3] = 1

        return results

    print('pred: ' + str([float(i.squeeze(0)) for i in y_test]))
    print('true: ' + str([float(i.squeeze(0)) for i in y_test_pred]))

    print('pred: ' + str(calculate_results(stock, classification_pred)))
    print('true: ' + str(calculate_results(stock, classification_true)))