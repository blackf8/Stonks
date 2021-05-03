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

for stock in stocks:
    
    #data prep
    key = stock
    df = pd.read_csv(key+".csv").fillna(0)
    test_df = pd.read_csv(key+"_extra.csv").fillna(0)
    train_data = df.append(test_df)
    train_ar = train_data['Close'].values
    history = [x for x in train_ar]
    predictions = list()
    last_val = train_ar[len(train_ar) - 1]
    test_result =  [str(key), 0,0,0,0,0]
    final_preds = [last_val]
    
    
    #ARIMA 
    for t in range(5):
        model = ARIMA(history, order=(5,2,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(yhat)
        final_preds.append(yhat.tolist()[0])

    test_result = classify(stock, last_val, predictions)
    print(final_preds)
    print(test_result)
    
   

