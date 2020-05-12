import matplotlib
matplotlib.use('Agg')
import random
import datetime as dt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tweepy
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
# import constants as ct
from Tweet import Tweet
import pandas_datareader.data as web
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import os.path
from sklearn.metrics import mean_squared_error
from os import path
from statsmodels.tsa.arima_model import ARIMA
import csv
from flask import Flask, render_template
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
# import mpld3
# import pylab
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# from matplotlib.figure import Figure
# import base64
# from io import BytesIO
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

consumer_key = os.environ['TWITTER_CONSUMER_KEY']
consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
access_token = os.environ['TWITTER_ACCESS_TOKEN']
access_secret = os.environ['TWITTER_ACCESS_SECRET']



num_of_tweets=int(20)


n=0
def get_stock_data(symbol, from_date, to_date):
    #data = yf.download(symbol, start=from_date, end=to_date)
    data=web.DataReader(symbol,data_source='yahoo',start=from_date,end=to_date)
    #data=web.DataReader(symbol,from_date,to_date,data_source='yahoo')
    df = pd.DataFrame(data=data)

    # df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    # df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # df = df[['Close', 'HighLoad', 'Change', 'Volume']]
    df.to_html('templates/stock_data.html')
    # with open('templates/stock_data.html','r') as f:
    #     with open('templates/newfile.html','w') as f2: 
    #         f2.write("""<form action="{{ url_for('home')}}" method="get">
    #   <button type="submits" class="button">Home</button>
    # </form>""")
    #         f2.write(f.read())
    # os.rename('templates/newfile.html','templates/stock_data.html')
    # with open('templates/stock_data.html','a') as f:
    #     f.write("""<form action="{{ url_for('predict')}}" method="get">
    #   <button type="submits" class="button">Home</button>
    # </form>""")
    return df



def test_stationarity(timeseries):
    
    
    #Determing rolling statistics
    
    rolmean=timeseries.rolling(window=20).mean()
    rolstd=timeseries.rolling(window=20).std()


    #Perform Dickey-Fuller test:
    # print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # print('ADF Statistic: %f ' % dftest[0])
    # print('p Statistic: %f ' % dftest[1])
    dfoutput = pd.Series(dftest[0:2], index=['open','high'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    # print(dfoutput)
    if(dftest[0]<dftest[4]["5%"]):
        # print("Reject H0, Time series is Stationary")
        return 1
    else:
        # print("Time series is not stationary")
        return 0


def make_data_stationary(tss):
    
    
    # print("Making Data Stationary....")
    # print("Log Transformations")
    ts_log = np.log(tss)

    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    # plt.plot(ts_log_diff)
    
    test_stationarity(ts_log_diff)
    
    return ts_log_diff


def inverse(p,df):
    
    tss=make_data_stationary(df['Close'])
    tss_log_diff=tss
    
    ts_log=np.log(df['Close'])
    
    tss_log_shift=ts_log.shift()
    tss_log_shift.dropna(inplace=True)
    
    x1=tss_log_shift[0]
    
    pi=[]
    
    pref=x1
    pi.append(x1)
    for i in range(len(p)):

        pi.append(p[i]+pref)
        pref=p[i]+pref

    ppi=np.exp(pi)
    
    return ppi
    
    
    

# tss=make_data_stationary(ts)


# # Forecasting

# In[109]:


def moving_average(df,w,seed_):
    
    tss=df['Close']
    ts=make_data_stationary(df['Close'])
    mad=0
    X = ts.values
    window = w
    pred=[]
    history=X[0:w]
    mse=0
    

    for i in range(len(X)-window+1):
        
        yhat=np.mean(X[i:i+w])
        mad+=abs(yhat-X[i])
        pred.append(yhat)
        
#         print('predicted=%f, expected=%f' % (yhat, X[i]))
    
    p=[]
    for i in range(len(history)):
        p.append(history[i])

    for i in range(len(pred)):
        p.append(pred[i])
    
    p_original=inverse(p,df)
    
     
    error = math.sqrt(mean_squared_error(tss, p_original[1:]))
    
    # print('Test RMSE: %.7f' % error)
    
    p=pd.DataFrame(p_original[1:],index=df.index)
    


    # fig = Figure(figsize=(5, 4), dpi=100)
    # A canvas must be manually attached to the figure (pyplot would automatically
    # do it).  This is done by instantiating the canvas with the figure as
    # argument.
    # canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    # ax = fig.add_subplot(111)
    # ax.plot(p,label="Predicted",color="blue")
    # ax.plot(df['Close'],label="Actual",color="red")
    # ax.legend()
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    # Option 1: Save the figure to a file; can also be a file-like object (BytesIO,
    # etc.).
    # fig.savefig("static/images/foo_ma"+str(n)+".png")
    # fig.clf()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.plot(df['Close'],label="Actual",color="red")
    plt.plot(p,label="Predicted",color="blue")
    plt.legend()
    plt.title('Moving Average RMSE : %.4f' % error)
    
   
    # plt.show()
    
    # zoom plot
    # plt.plot(X[w:100],label="Actual")
    # plt.plot(pred[w:100], color='blue',label="Predicted")
    # plt.legend()
    # plt.show()
    global n
    # import os
    # for i in range(n):
    #     if os.path.exists('static/images/foo_ma'+str(i)+'.png'):
    #         os.remove('static/images/foo_ma'+str(i)+'.png')
    random.seed(seed_)
    plt.savefig('static/images/foo_ma'+str(random.randint(1,1e9))+'.png',bbox_inches = "tight")
    plt.close()
    # num_fig+=1
#     history.append(pred)
    #history=np.append(history,pred)
    return df['Close'].values[-1],p_original[-1],error


def exponential_smoothing(df,alpha,seed_):
    
    ts=df['Close']
    X = ts.values
    forecasts=[]
    forecasts.append(X[0])
    mse=0.0
    mad=0
    random.seed(seed_)
    for i in range(1,len(X)):
        
        f=alpha*X[i-1] + (1-alpha)*forecasts[i-1]
        
        mse+=(f-X[i])**2
        mad+=abs(f-X[i])
        forecasts.append(f)
    
    mse=mse/len(X)
    mad=mad/len(X)
    rmse=math.sqrt(mse)
    # print("Mean Squared Error : ",mse)
    # print("Mean Absolute Deviation : ",mad)
    # print("Root Mean Squared Error : ",rmse)
    #plt.figure(figsize=(16,8))
    
    f=forecasts
    f=pd.DataFrame(f,index=ts.index)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.plot(ts, label='Actual',color='red')
    plt.plot(f, label='Predicted',color='blue')
    plt.title('Exponential Smoothing Moving Average RMSE : %.4f' % rmse)
    plt.legend()
    # plt.show()
    
    global n
    plt.savefig('static/images/foo_es'+str(random.randint(1,1e9))+'.png',bbox_inches = "tight")
    plt.clf()
    # num_fig+=1
    # plt.legend(loc='best')
    # plt.show()
    return (X[len(X)-1],forecasts[len(forecasts)-1],rmse)

# In[95]:


# exponential_smoothing(tss,0.2)


# In[107]:


def ARIMA_forecast(df,seed_):
    
    tss=df['Close']
    ts_log=np.log(df['Close'])
    ts=make_data_stationary(df['Close'])
#     ts=ts.values
    model=ARIMA(ts,order=(2,1,0))
    results_AR=model.fit(disp=-1)
    pred=results_AR.fittedvalues
#     print(len(pred))
#     plt.plot(ts.values,color='blue')
#     plt.plot(pred.values,color='red')
#     mse_s=sum((pred.values-ts.values[1:])**2)/len(pred)
#     plt.title('MSE : %.4f' % mse_s)

    predictions_ARIMA_diff=pd.Series(results_AR.fittedvalues, copy=True)
    
    predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
    # print(predictions_ARIMA_diff_cumsum.head())
#     print(len(predictions_ARIMA_diff_cumsum))
    prediction_ARIMA_log=pd.Series(ts_log)
    
    prediction_ARIMA_log=prediction_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    
    predictions_ARIMA=np.exp(prediction_ARIMA_log)
    pa=pd.DataFrame(predictions_ARIMA,index=df.index)
    plt.plot(predictions_ARIMA,color='red',label='Predicted')
    plt.plot(df['Close'],color='blue',label='Actual')
#     print(len(predictions_ARIMA),len(df['Close']))
    mse=(sum((predictions_ARIMA.values-df['Close'].values)**2))/len(predictions_ARIMA)
    rmse=math.sqrt(mse)
    # print('ARIMA RMSE : %.4f' % mse)
    # plt.title('ARIMA RMSE : %.4f' % rmse)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    global n
    random.seed(seed_)
    plt.savefig('static/images/foo_arima'+str(random.randint(1,1e9))+'.png',bbox_inches = "tight")
    plt.clf()
    # plt.legend()
    # plt.show()
    
    return (df['Close'].values[-1],predictions_ARIMA[-1],rmse)

def LSTM_(df,seed_):
    
    ts=df['Close']
    ts=pd.DataFrame(ts)
    data=ts
    look_back=int(len(data)*0.02)
    sc = MinMaxScaler(feature_range = (0, 1))
    data_scaled = sc.fit_transform(data)

    # Creating a data structure with 60 timesteps and 1 output
    x = []
    y = []
    for i in range(look_back, len(data)):
        x.append(data_scaled[i-look_back:i, 0])
        y.append(data_scaled[i, 0])
    x, y = np.array(x), np.array(y)

    # Reshaping
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=False)
    model = Sequential()
    # Adding the first LSTM layer
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    # model.add(Dropout(0.2))

    # Adding a second LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = 1))
    
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    # Iterations is the number of batches needed to complete one epoch.
    model.fit(x_train, y_train, epochs = 50, batch_size = 32)
    
    pred=model.predict(x_test)
    
    pred = sc.inverse_transform(pred)
    
    x_test_price=data_scaled[len(data_scaled)-len(x_test):]
    
    actual_price=sc.inverse_transform(x_test_price)
    
    test_len=len(actual_price)
    train_len=len(ts)-test_len
    train=ts[:train_len]
    test=ts[train_len:]
    
    result=train
    result=np.append(result,pred)
    
    result=pd.DataFrame(result,index=ts.index)
    
    error=mean_squared_error(ts,result)
    rmse=math.sqrt(error)
    random.seed(seed_)
    plt.plot(result, color = 'red', label = 'Predicted Stock Price')
    plt.plot(ts, color = 'blue', label = 'Real Stock Price')
    
    plt.title('Stock Price Prediction Using LSTM RMSE : %.4f' % rmse)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    global n
    plt.savefig('static/images/foo_lstm'+str(random.randint(1,1e9))+'.png',bbox_inches = "tight")
    plt.clf()
    
    return (actual_price[-1],pred[-1],rmse)

    
def retrieving_tweets_polarity(symbol):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    user = tweepy.API(auth)

    tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en').items(num_of_tweets)

    tweet_list = []
    # b="""<form action="{{ url_for('predict')}}" method="get">
    #   <button type="submits" class="button">Home</button>
    # </form>"""
    # tweet_list.append(b)
    # tweet_list.append("<link rel='stylesheet' 
    #     href='{{ url_for("static",filename="style2.css")}}'>")
    s='\"stylesheet\"'
    l='\"{{ url_for(\'static\',filename=\'style2.css\')}}\">'
    tweet_list.append('<link rel='+ s + ' href=' + l)
    
    global_polarity = 0
    for tweet in tweets:
        tweet_list.append("<div class='tweet-body'>")
        tweet_list.append("<pre>")
        tw = tweet.full_text
        # print(tw)
        blob = TextBlob(tw)
        polarity = 0
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            global_polarity += sentence.sentiment.polarity
        tweet_list.append(tw)
        tweet_list.append("</pre>")
        tweet_list.append("</div>")
        
    
    global_polarity = global_polarity / len(tweet_list)
    np.savetxt('templates/tweets.html',tweet_list,newline='\n',fmt="%s",encoding="utf-8")
    return global_polarity


def polarity(symbol):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    user = tweepy.API(auth)

    tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en').items(num_of_tweets)
    tweet_list = []
    s='\"stylesheet\"'
    l='\"{{ url_for(\'static\',filename=\'style2.css\')}}\">'
    tweet_list.append('<link rel='+ s + ' href=' + l)
    
#     global_polarity = 0
    pos_count=0
    neg_count=0
    net_count=0
    
    sid_obj = SentimentIntensityAnalyzer() 
    
    for tweet in tweets:
        tw = tweet.full_text

        tweet_list.append("<div class='tweet-body'>")
        tweet_list.append("<pre>")

        # print(tw)
        tweet_list.append(tw)
        sentiment_dict = sid_obj.polarity_scores(tw)
        
        tweet_list.append(tw)
        tweet_list.append("</pre>")
        tweet_list.append("</div>")
        
        if sentiment_dict['compound'] >= 0.05 : 
            # print("Positive")
            pos_count+=1 

        elif sentiment_dict['compound'] <= - 0.05 : 
            # print("Negative") 
            neg_count+=1

        else : 
            # print("Neutral") 
            net_count+=1
    
    np.savetxt('tweet_file.html',tweet_list, fmt='%s',encoding="utf-8")
    if pos_count>neg_count and pos_count>net_count:
        # print("Net Sentiment is Positive")
        return 1
    else: 
        # print("Net Sentiment is Negative")
        return -1


def stock_forecasting(ts,algorithm,seed_):
    
    if(algorithm=="MOVING AVERAGE"):
        ma=moving_average(ts,3,seed_)
        return (ma[0],ma[1])
    
    elif(algorithm=="EXPONENTIAL SMOOTHING"):
        es=exponential_smoothing(ts,0.2,seed_)
        return (es[0],es[1])
    
    elif(algorithm=="ARIMA"):
        af=ARIMA_forecast(ts,seed_)
        return (af[0],af[1])
    
    elif(algorithm=="LSTM"):
        l=LSTM_(ts,seed_)
        return (l[0],l[1])
    
    else:
         # In case No Algorithm is specified, return the results of the Algorithm with the Least Error
        # print("Running all Algorithms ....")

        A=moving_average(ts,3,seed_)
        B=exponential_smoothing(ts,0.2,seed_)
        C=ARIMA_forecast(ts,seed_)
        D=LSTM_(ts,seed_)
        
        mini=min(A[2],min(B[2],min(C[2],D[2])))
        
        if(mini==A[2]):
            print("Using Moving Average .....")
            return (A[0],A[1])
        
        elif(mini==B[2]):
            print("Using Exponential Smoothing .....")
            return (B[0],B[1])
        
        elif(mini==C[2]):
            print("Using ARIMA .....")
            return (C[0],C[1])
        else:
            print("Using LSTM .....")
            return (D[0],D[1])


# In[122]:


def recommending(symbol,start_date,end_date,m,algorithm="default"):
    
    global n
    n=m
    # Get stock data
    df=get_stock_data(symbol, start_date, end_date)

    # Forecast the data

    seed_=random.randint(1,1e9)

    actual,forecast=stock_forecasting(df,algorithm,seed_)

    # Compute global popularity of stock
    # global_polarity=retrieving_tweets_polarity(symbol)
    sentiment=polarity(symbol)
    # Final Analysis
    s=""
    if(actual < forecast):
        if sentiment > 0:
            s="According to the predictions and twitter sentiment analysis -> Investing in %s is a GREAT idea!" % str(symbol)
        elif sentiment < 0:
            s="According to the predictions and twitter sentiment analysis -> Investing in %s is a BAD idea!" % str(symbol)
    else:
        s="According to the predictions and twitter sentiment analysis -> Investing in %s is a BAD idea!" % str(symbol)

    return s,seed_
# In[123]:
# recommending('GOOG','3/12/2019','31/3/2020')

