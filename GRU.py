import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import math, time
from sklearn.metrics import mean_squared_error


stocks = ['VZ', 'T', 'WMT', 'MGM', 'GPS', 'GT', 'BBY', 'AFG', 'ERJ', 'MYE', 'ECPG', 'GCO', 'MPC', 'TRI', 'UFI']

for stock in stocks:
    data = pd.read_csv("data/{}.csv".format(stock))
    price = data[['Close']].fillna(method='ffill')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

    def split_data(stock, lookback):
        data_raw = stock.to_numpy() # convert to numpy array
        data = []
        
        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - lookback): 
            data.append(data_raw[index: index + lookback])
        
        data = np.array(data)
        test_set_size = 5
        train_set_size = data.shape[0] - (test_set_size)
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        
        return [x_train, y_train, x_test, y_test]


    lookback = 20 # choose sequence length
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    '''print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)'''


    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)


    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    class GRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(GRU, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn) = self.gru(x, (h0.detach()))
            out = self.fc(out[:, -1, :]) 
            return out

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


    hist = np.zeros(num_epochs)
    start_time = time.time()
    gru = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_gru)
        #if t % 10 == 0:
            #print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time()-start_time    
    #print("Training time: {}".format(training_time))


    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_gru.detach().numpy())

    '''# calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    gru.append(trainScore)
    gru.append(testScore)
    gru.append(training_time)'''

    '''
    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(data[len(data)-len(y_test):].index, y_test, color = 'red', label = 'Real Stock Price')
    axes.plot(data[len(data)-len(y_test_pred):].index, y_test_pred, color = 'blue', label = 'Predicted Stock Price')
    #axes.xticks(np.arange(0,394,50))
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