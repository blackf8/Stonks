import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import collections
import matplotlib.dates as mdates
import matplotlib.cbook as cbook


#Symmetric mean absolute percentage error
def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) +       np.abs(y_true))))

#Classifies the data, given the raw values
def classify(stock, last_val, data):
    result = [stock, 0, 0, 0, 0, 0]
    for i, day in enumerate(data):
        two_percent = last_val*0.02
        five_percent = last_val*0.05

        if(day >= last_val + five_percent):
            result[5] = 1
            result[4] = 1
        elif(day >= last_val + two_percent and day < last_val + five_percent):
            result[4] = 1
        elif(day <= last_val - two_percent and day > last_val - five_percent):
            result[2] = 1
        elif(day <= last_val - five_percent):
            result[1] = 1
            result[2] = 1
        
    if 1 not in result:
            result[3] = 1
        
    return result



#Main Algorithm  
stocks = ['VZ', 'T', 'WMT', 'MGM', 'GPS', 'GT', 'BBY', 'AFG', 'ERJ', 'MYE', 'ECPG', 'GCO', 'MPC', 'TRI', 'UFI']
stock_arrays = []
total_MSE = 0
total_SMAPE = 0
for stock in stocks:
    #data prep
    key = stock
    df = pd.read_csv(key+".csv").fillna(0)
    test_df = pd.read_csv(key+"_extra.csv").fillna(0)
    train_data = df.append(test_df[1:6])
    test_data = test_df[6:]
    train_ar = train_data['Close'].values
    test_ar = test_data['Close'].values
    history = [x for x in train_ar]
    predictions = list()
    test_result =  [str(key), 0,0,0,0,0]
    last_val = train_ar[len(train_ar) - 1]
    print("Last " + str(last_val))
    true_result = classify(stock,last_val, test_ar)

    
    #ARIMA
    for t in range(len(test_ar)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_ar[t]
        history.append(obs)

    test_result = classify(stock, last_val, predictions)
    stock_arrays.append((test_result, true_result))
    error = mean_squared_error(test_ar, predictions)
    total_MSE += error
    error2 = smape_kun(test_ar, predictions)
    total_SMAPE += error2
    


    print("RESULTS: " + key)
    print('Testing Mean Squared Error: %.3f' % error)
    print('Symmetric mean absolute percentage error: %.3f' % error2)
    print("last value: " +str(last_val))
    print()
    print("predicted stock prices: " + str(plot_pred))
    print("True stock prices: " + str(test_ar.tolist()))
    print("predicted array: " + str(test_result))
    print("True array: " + str(true_result))
    
   


    #plot data
    fig = plt.figure(figsize=(12,7))
    ax = plt.axes()
    ax.plot(test_data["Date"], test_data["Close"],color='red', label='Actual Price')
    ax.plot(test_data["Date"], plot_pred, color='blue', label='Predicted Price')
    ax.scatter(test_data["Date"], test_data["Close"])
    ax.scatter(test_data["Date"], plot_pred)
    plt.title('Prices Prediction')
    plt.xlabel('Dates')
    plt.ylabel(key + ' Prices')
    plt.legend()


    #Used for result data porcessing
    """
    f = open("ARIMAresults.txt", "a")
    f.write("\n")
    f.write("pred: " + str(plot_pred) + "\n")
    f.write("true: " + str(test_ar.tolist()) + "\n")
    f.write("pred: " + str(test_result) + "\n")
    f.write("true: " + str(true_result) + "\n")
    f.close()
    """

#calculate percent error
correct = 0
incorrect = []
for pair in stock_arrays:
    if collections.Counter(pair[0]) == collections.Counter(pair[1]):
        correct += 1
    else:
        incorrect.append(pair)

print("Percent error: " + str((correct/15)*100))
print("Incorrect: " +str(incorrect))
print("Average MSE: " +str(total_MSE/15))
print("Average SMAPE: " +str(total_SMAPE/15))