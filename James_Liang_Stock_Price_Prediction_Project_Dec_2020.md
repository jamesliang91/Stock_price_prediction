# Capstone Project: Stock Price Prediction
*Author: Minghan Liang (James),  Dec. 2020*


### Problem Statement
In this project, we will aim to predict the adjusted close price of stocks listed in the U.S. market. This project's objective can be summarized as a supervised learning problem where inputs are raw, historical data of a particular stock acquired through Yahoo! Finance API and output are numerical values of adjusted close price of the chosen stock in the next day.

### Workflow
The solution in this python Jupyter notebook script will consist of the following sections:
  1. Initialization of required packages and tools
  2. Acquisition, storage, loading, and formatting of data
  3. Visualization of data 
  4. Feature Engineering methods
  5. Data set preparation for model training and testing  
  6. Construction of the proposed model, training, and testing with evaluation metric
  7. Comparison of the proposed model with benchmark models

## Section 1: Initialization
In this section, we will list various python and anaconda packages that will be used later throughout this project and define multiple helper functions that perform data extractions and saving through popular stock data API

### 1.1 Package initialization
Here we will import common toolkits for scientific computing tasks, including:
1. **common usage tools**:  os / datetime 
2. **visualization tools**:  matplotlib.pyplot / mpl_finance
3. **scientific computing tools**:  pandas / numpy
4. **machine learning packages**:  Scikit-learn / Tensorflow & Keras / Xgboost
5. **stock data API & technical indicator calculation tools**:  pandas_datareader / talib


```python
# import basic packages for scientific computing and visualizations
import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.dates as mdates 
```


```python
# pandas_datareader.data enables downloading of stock data from multiple API
import pandas_datareader.data as web  

# mpl_finance is a package used to draw candle stick graph, a popular genre of stock data visualization 
import mpl_finance as mpl 

# import talib for calculation of technical indicators of stock data 
import talib
```

    /Users/jamesliang/anaconda3/lib/python3.7/site-packages/mpl_finance.py:22: DeprecationWarning: 
    
      =================================================================
    
       WARNING: `mpl_finance` is deprecated:
    
        Please use `mplfinance` instead (no hyphen, no underscore).
    
        To install: `pip install --upgrade mplfinance` 
    
       For more information, see: https://pypi.org/project/mplfinance/
    
      =================================================================
    
      category=DeprecationWarning)



```python
# Import preprocessing module and multiple error functions from Scikit-Learn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
```


```python
# Import Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
```


```python
# Import XGBoost 
import xgboost as xgb
```

### 1.2: Define data import & saving methods
Here we will define functions that allow us to do the followings:
1. Acquire formatted stock data from popular stock data API and stored data locally in CSV files. In this project, we use **yahoo! finance** API 
2. Import stored data and make adjustments on the format for later usage


```python
# Download data through API 

def Get_And_Save_Data(ticker, data_dir, start_date = '2000-01-01', end_date = dt.date.today()):
    print ('import data: '+ticker)
    data = web.DataReader(ticker, 'yahoo', start_date, end_date)
    data_file_name= ticker + '_full_data_' + str(end_date) + '.csv'
    print (data_file_name)
    
    # check existence of data dir, if not exist then create one
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    data.to_csv(os.path.join(data_dir, data_file_name))
    
    print ('data directory: ' + data_dir)
    print ('data file name: ' + data_file_name)
    return (data)

# Import downloaded data 

def Import_And_Clean_Csv_Data(data_dir, filename):
    data = pd.read_csv(os.path.join(data_dir, filename), header=0)
    
    data.reset_index(inplace=True)
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    data = data[cols]
    
    # fill Null & NAN entry, use forward filling to prevent data leakage
    data.fillna(method = 'ffill', inplace = True )
    return (data)

```

In this project, we use three stocks as example: 
1. **Amazon.com, INC** (ticker: AMZN): Technology industry / E-commerce & Cloud Service / High stock price value / fast growth trend
2. **JPMorgan Chase & Co** (ticker: JPM): Financial Service industry / Commercial & Investment bank / medium stock price value /  slow growth trend
3. **Ford Motor Company** (ticker: F):  Manufacturing industry / Automobile Manufacturer / low stock price value / flat trend


```python
# Create a folder named 'stock_price_forecast_data' and download raw stock data into the folder
start_date = dt.date(2010,1,1)
end_date = dt.date(2020,12,20)

Get_And_Save_Data(ticker = 'AMZN',
                  data_dir = 'stock_price_forecast_data',
                  start_date = start_date,
                  end_date = end_date)
```

    import data: AMZN
    AMZN_full_data_2020-12-20.csv
    data directory: stock_price_forecast_data
    data file name: AMZN_full_data_2020-12-20.csv





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-12-31</th>
      <td>137.279999</td>
      <td>134.520004</td>
      <td>137.089996</td>
      <td>134.520004</td>
      <td>4523000</td>
      <td>134.520004</td>
    </tr>
    <tr>
      <th>2010-01-04</th>
      <td>136.610001</td>
      <td>133.139999</td>
      <td>136.250000</td>
      <td>133.899994</td>
      <td>7599900</td>
      <td>133.899994</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>135.479996</td>
      <td>131.809998</td>
      <td>133.429993</td>
      <td>134.690002</td>
      <td>8851900</td>
      <td>134.690002</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>134.729996</td>
      <td>131.649994</td>
      <td>134.600006</td>
      <td>132.250000</td>
      <td>7178800</td>
      <td>132.250000</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>132.320007</td>
      <td>128.800003</td>
      <td>132.009995</td>
      <td>130.000000</td>
      <td>11030200</td>
      <td>130.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-12-14</th>
      <td>3190.469971</td>
      <td>3126.000000</td>
      <td>3143.000000</td>
      <td>3156.969971</td>
      <td>4155800</td>
      <td>3156.969971</td>
    </tr>
    <tr>
      <th>2020-12-15</th>
      <td>3188.500000</td>
      <td>3130.489990</td>
      <td>3181.010010</td>
      <td>3165.120117</td>
      <td>3323700</td>
      <td>3165.120117</td>
    </tr>
    <tr>
      <th>2020-12-16</th>
      <td>3247.000000</td>
      <td>3163.679932</td>
      <td>3176.010010</td>
      <td>3240.959961</td>
      <td>4427600</td>
      <td>3240.959961</td>
    </tr>
    <tr>
      <th>2020-12-17</th>
      <td>3263.510010</td>
      <td>3221.000000</td>
      <td>3250.000000</td>
      <td>3236.080078</td>
      <td>3474300</td>
      <td>3236.080078</td>
    </tr>
    <tr>
      <th>2020-12-18</th>
      <td>3249.419922</td>
      <td>3171.600098</td>
      <td>3243.989990</td>
      <td>3201.649902</td>
      <td>5995700</td>
      <td>3201.649902</td>
    </tr>
  </tbody>
</table>
<p>2762 rows × 6 columns</p>
</div>



## Section 2: Feature engineering and visualizations
In this section we will define functions that perform feature engineering on original stock data, and construct a few functions for data visualizations

### 2.1 Feature engineer 
In this project, we will use raw stock data to calculate some technical indicators that reveals important features of a stock, including the following: 

1. **Price average indicator(s)**: Indicate the average historical performance of a stock within a given time window. Price average allows the model to capture information regarding the growth speed of stocks within a market circumstance (i.e., in most cases, we saw price average goes up overtime except in financial crisis, which reveals the fact that stock assets grow in value as the economy grows at a stable rate)
2. **Transaction behavior & trend indicator(s)**: Reveal some information regarding market participants' behaviors, along with the price status of stocks resulting from those behaviors.
3. **Volume Indicator**: Reveals essential information regarding the market conditions (the volume can be used to infer important information on the industrial sector/trading market / geological region of the underlying stock. That information can reveal systematic factors rather than idiosyncratic ones.
4. **Hilbert Transformation indicator(s)**: A set of technical indicators backed by exclusive mathematical theories on signal processing, where the stock price is viewed as a sequence that possesses hidden periodicity


```python
def Feature_value_scaler(feature_data,scaling_list):
    base_value_dict = {}
    for feature in scaling_list:
        first_value = feature_data[feature].iloc[0]
        base_value_dict[feature] = first_value
        if first_value == 0:
            continue
        else:
            feature_data[feature] = (feature_data[feature]/first_value)
            
    return feature_data, base_value_dict   
    
def Feature_Engineer(data): 
    # param -- data: the dataframe of underlying stock 
    
    # In this project, we set our time window for calculating various price average indicators to be 30
    time_period = 30
    
    # log return 
    output = data.copy(deep = True) 
    
    #if data['Date']
    data_datetime_str_to_date = output['Date'].values
    for i in range (len(data_datetime_str_to_date)):
        data_datetime_str_to_date[i] = dt.datetime.strptime(data_datetime_str_to_date[i], '%Y-%m-%d').date()
        
    output['Date'] = data_datetime_str_to_date    
    output['Log Return'] = np.log(output['Adj Close'] / output['Adj Close'].shift(1))
    (output['Log Return'])[0] = 0 # set the first entry of log return to 0
    
# USING TA-lib to calculate various technical indicators of stock 

# Price Average indicators:
    # 1. SMA -- Simple Moving Average : arithmetic average, less reactive to recent information
    output['TA_SMA'] = talib.SMA(output['Adj Close'].values, time_period)
    
    # 2. EMA -- Exponential Moving Average: place more weight on recent price, more reactive to recent information
    output['TA_EMA'] = talib.EMA(output['Adj Close'].values, time_period)

    # 3. MACD: Components 
    # 3.1. DIF = 12-day EMA - 26-day EMA: DIF > 0 indicate short-term price in upward trend 
    # 3.2. DEA = Moving Average of DIF (typical period: 9-day):
    # Greater Absolute value of DEA indicate greater openning of DIF, which means an greater
    # acceleration of upward or downward movement of stock price

    output['TA_MACD'], output['TA_MACDSIGNAL'], output['TA_MACDHIST'] = talib.MACDFIX(output['Adj Close'].values)
    
# trasaction behavior & trend indicators: 

    # 4. RSI -- relative strength index: detect upward and downward price movement from past period to 
    # capture possible transaction trend (i.e. long vs short)
    
    output['TA_RSI'] = talib.RSI(output['Adj Close'].values, time_period)
    
    # 5. MOM -- momentum 
    output['TA_MOM'] = talib.MOM(output['Adj Close'].values, time_period)
    
    # 6. KDJ -- KDJ index: stochastic indicators that captures long & short trend from High, Low and Close price    
    output['KDJ_K'], output['KDJ_D'] = talib.STOCH(output['High'].values, 
                                                               output['Low'].values,
                                                               output['Close'].values,
                                                               fastk_period=9,
                                                               slowk_period=3,
                                                               slowk_matype=0,
                                                               slowd_period=3,
                                                               slowd_matype=0)    
 
    # 7. WILLR: William's R% -- detecting trend (overbrought vs oversold)
    output['TA_WILLR'] = talib.WILLR(output['High'].values, output['Low'].values, output['Close'].values, time_period)
    
    
# Volumn Indicators: 

    # 8. OBV -- On balance Volumn: capture  "kinetic energy" level of a stock
    # Notice: need to convert Volumn from 'numpy.int64' to 'numpy.float64'
    output['TA_OBV'] = talib.OBV(output['Adj Close'].values, np.asarray(output['Volume'].values, dtype='float'))

    
# special features: Hilbert Transformation indicator:

    # 9. T_DCPERIOD: Hilbert Transform (Dominant Cycle Period) 
    output['TA_HT_DCPERIOD'] = talib.HT_DCPERIOD(output['Close'].values)

    # 10. T_TRENDMODE: Hilbert Transform (Trend vs Cycle Mode)s
    output['TA_HT_TRENDMODE'] = talib.HT_TRENDMODE(output['Close'].values)
    
    output = output.dropna()
    scaling_list = ['Adj Close', 'TA_SMA', 'TA_EMA', 'TA_MACD', 'TA_MACDSIGNAL','TA_MACDHIST',\
            'TA_RSI','TA_MOM','KDJ_K','KDJ_D','TA_WILLR','TA_OBV','TA_HT_DCPERIOD']  
    output, output_base_value_dict = Feature_value_scaler(output,scaling_list)
    
    cols = ['Date', 'Adj Close', 'Log Return', 'TA_SMA', 'TA_EMA', 'TA_MACD', 'TA_MACDSIGNAL','TA_MACDHIST',\
            'TA_RSI','TA_MOM','KDJ_K','KDJ_D','TA_WILLR','TA_OBV','TA_HT_DCPERIOD','TA_HT_TRENDMODE']
    output = output[cols]
    
    return output, output_base_value_dict
```


```python
AMZN_data = Import_And_Clean_Csv_Data('stock_price_forecast_data','AMZN_full_data_2020-12-20.csv')
AMZN_features_data, AMZN_base_value_dict = Feature_Engineer(AMZN_data)
```

    /Users/jamesliang/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:29: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


### 2.2 Error function: Mean Absolute Percentage Error
Here we define the error function that calculates the mean absolute percentage error. Later this function will be used in the error evaluation section. Note that we implement an if-else statement to eliminate the division by zero problem, though it rarely occurs in stock price records. 


```python
# define mean_absolute_percentage_error evaluation function
def mean_absolute_percentage_error(real_val, prediction_val):
    assert (len(real_val) == len(prediction_val)),"number of real values does not match number of prediction values!"
    real_val = np.asarray(real_val)
    prediction_val = np.asarray(prediction_val)
    diff = np.subtract(real_val, prediction_val)
    mape_error = 0
    for i,j in zip(diff,real_val):
        if j == 0:
            continue
        else:
            mape_error += abs(i/j)
    mape_error = mape_error / len(prediction_val)
    return mape_error

```

### 2.3 Visualization
Construct a few functions that help us to visualize various indicators


```python
# visualization tools

# candle stick graph: recommend only use 1-year data
def candlestick_graph(data):
    candle_data = data[:]
    
    # candlestick_ohlc function requires datetime object to be transformed into integers
    # transform "Date" column into number format (first from str to datetime, then from datetime to number)
    
    candle_data['Date'] = mdates.date2num(candle_data['Date'])     
    fig,ax = plt.subplots(figsize = (1200/72,480/72))
    mpl.candlestick_ohlc(ax, candle_data.values, width=0.4, colorup='#77d879', colordown='#db3f3f')
    ax.xaxis_date()
    plt.title("Candle Stick Graph of Close Price")
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.show()
    
    
# visualization of price average indicators
def Average_graph(data):
    fig,ax = plt.subplots(figsize = (1200/72,480/72))
    
    plt.plot(data['Date'],data['Adj Close'],'k',lw = 1.0,label = 'Adj Close')
    plt.plot(data['Date'],data['TA_SMA'],'g',lw = 1.0,label = 'TA_SMA')
    plt.plot(data['Date'],data['TA_EMA'],'r',lw = 1.0,label = 'TA_EMA')
    plt.legend(loc=0)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Adjusted Close Price and Average Lines')
    plt.show()

# visualization of trend indicators
def trend_indicator_graph(data):
    fig,ax = plt.subplots(figsize = (1200/72,480/72))

    plt.plot(data['Date'],data['TA_RSI'],'y',lw = 1.0,label = 'TA_RSI')
    plt.plot(data['Date'],data['KDJ_K'],'g',lw = 1.0,label = 'KDJ_K')
    plt.plot(data['Date'],data['KDJ_D'],'r',lw = 1.0,label = 'KDJ_D')
    plt.plot(data['Date'],data['TA_WILLR'],'k',lw = 1.0,label = 'TA_WILLR')
    plt.legend(loc=0)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title("Price Trend Indicators")
    plt.show()

```


```python
candlestick_graph(AMZN_data.iloc[-200:,:])
```


    
![png](output_21_0.png)
    



```python
Average_graph(AMZN_features_data.iloc[-200:,:])
```


    
![png](output_22_0.png)
    



```python
trend_indicator_graph(AMZN_features_data.iloc[-100:,:])
```


    
![png](output_23_0.png)
    


## Section 3: Construct Training and Testing Dataset
After data initialization and feature engineering process, we will define a function that performs train-test split and transformation. The goal is to prepare data in the proper format for our primary model (RNN with LSTM) and benchmark model (XGBoost regressor)
### Methodologies:
The subject of stock price forecasting can be framed as a supervised machine learning problem: constructing a value predictor, using some historical data to train the predictor, and the rest of the data to perform testing. Notice that in time series forecasting, we will strictly preserve the order of data to prevent data leakage.

An RNN input unit is a "sequence." In stock prediction, it is a vector consisting of multiple days' featured stock data. The length of such a "sequence" equals the sum of **train window length** and **prediction window length**.

### Core concepts in this section:
1. **train window length** ( As *window_len* ): The number of days used as time window to extract information is the **window length** (i.e.: how many days of featured stock data will we use in one step to predict stock price?). In this project, we will set the default value as 10 (trading days), which is roughly equivalent to the length of two trading weeks.
2. **prediction window length** ( As *output_step* ): the number of future days on which we want to predict stock price (i.e.: how many days of stock's adjusted close price that we want to predict in one step, given data of training window length?). In this project, we will set the default value as 1 (trading day).



```python

def train_test_prep(data, window_len= 10, output_step=1, dropnan=True):
    # function parameter explanations:
    # -- data: featured stock data
    # -- window_len: as train window length, default = 10
    # -- output_len: as prediction window length, default = 1
    # -- split_ratio: ratio of train-test split, default = 0.75
    # -- dropnan: whether drop data entry with NaN value, default = True

    input_dim = 1 if type(data) is list else data.shape[1]
    data_df = pd.DataFrame(data)
    cols = list()
    col_names = list()
    
    # input sequence (t-n, ... t-1)
    for i in range(window_len, 0, -1):
        cols.append(data_df.shift(i))
        col_names += [('var%d(t-%d)' % (j+1, i)) for j in range(input_dim)]
        
    # forecast sequence (t, t+1, ... t+n), name columns by variable index and date 
    for i in range(0, output_step):
        cols.append(data_df.shift(-i))
        if i == 0:
            col_names += [('var%d(t)' % (j+1)) for j in range(input_dim)]
        else:
            col_names += [('var%d(t+%d)' % (j+1, i)) for j in range(input_dim)]
            
    # concatenate all data
    stacked_data = pd.concat(cols, axis=1)
    stacked_data.columns = col_names
    
    # drop rows with NaN values
    if dropnan:
        stacked_data.dropna(inplace=True)

    return stacked_data



def train_test_split (stacked_data, split_ratio = 0.75):
    # split data into training & test set, seperate input from output
    data_length = stacked_data.shape[0]
    split_index = int(round(data_length * split_ratio))
    
    train_set = stacked_data.values[ :split_index, :]
    test_set = stacked_data.values[split_index: , :]
    
    return train_set, test_set



def train_test_reformat(train_set, test_set, window_len= 10, output_step=1):
    
    input_dim = 1 if type(train_set) is list else int((train_set.shape[1]/(window_len + 1)))
    
    n_obs = input_dim * window_len
    train_X, train_y = train_set[:, :n_obs], train_set[:, -(input_dim*output_step)::input_dim]
    test_X, test_y = test_set[:, :n_obs], test_set[:, -(input_dim*output_step)::input_dim]
    train_X = train_X.reshape((train_X.shape[0], window_len, input_dim))
    test_X = test_X.reshape((test_X.shape[0], window_len, input_dim))
    
    return train_X, test_X, train_y, test_y

```

## Section 4: Construct primary model, training and testing
In this section, we construct a Recurrent Neural Network (RNN) model with LSTM (Long-Short Term Memory) architecture as our primary model. We will need to build and define various components in this section:
1. **Model Hyperparameters**: Size of input tensor / Size of output tensor / Number of hidden LSTM layer / Number of neurons (in hidden LSTM layer) / Learning rate (for optimization)

### 4.1: Load and prepare train & test data
As our first step, we will load our data (we use Amazon stock data as an example in this notebook).


```python
AMZN_data = Import_And_Clean_Csv_Data('stock_price_forecast_data','AMZN_full_data_2020-12-20.csv')
AMZN_features_data, AMZN_base_value_dict = Feature_Engineer(AMZN_data)
```

    /Users/jamesliang/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:29: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy



```python
AMZN_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-12-31</td>
      <td>137.089996</td>
      <td>137.279999</td>
      <td>134.520004</td>
      <td>134.520004</td>
      <td>4523000</td>
      <td>134.520004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-01-04</td>
      <td>136.250000</td>
      <td>136.610001</td>
      <td>133.139999</td>
      <td>133.899994</td>
      <td>7599900</td>
      <td>133.899994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-01-05</td>
      <td>133.429993</td>
      <td>135.479996</td>
      <td>131.809998</td>
      <td>134.690002</td>
      <td>8851900</td>
      <td>134.690002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-01-06</td>
      <td>134.600006</td>
      <td>134.729996</td>
      <td>131.649994</td>
      <td>132.250000</td>
      <td>7178800</td>
      <td>132.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-01-07</td>
      <td>132.009995</td>
      <td>132.320007</td>
      <td>128.800003</td>
      <td>130.000000</td>
      <td>11030200</td>
      <td>130.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
AMZN_features_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Adj Close</th>
      <th>Log Return</th>
      <th>TA_SMA</th>
      <th>TA_EMA</th>
      <th>TA_MACD</th>
      <th>TA_MACDSIGNAL</th>
      <th>TA_MACDHIST</th>
      <th>TA_RSI</th>
      <th>TA_MOM</th>
      <th>KDJ_K</th>
      <th>KDJ_D</th>
      <th>TA_WILLR</th>
      <th>TA_OBV</th>
      <th>TA_HT_DCPERIOD</th>
      <th>TA_HT_TRENDMODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>2010-02-19</td>
      <td>1.000000</td>
      <td>-0.004754</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2010-02-22</td>
      <td>1.004170</td>
      <td>0.004161</td>
      <td>0.996734</td>
      <td>0.997442</td>
      <td>0.936218</td>
      <td>0.950496</td>
      <td>1.008972</td>
      <td>1.016002</td>
      <td>0.813985</td>
      <td>1.180908</td>
      <td>0.989000</td>
      <td>0.969678</td>
      <td>0.942171</td>
      <td>1.076906</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2010-02-23</td>
      <td>0.997617</td>
      <td>-0.006546</td>
      <td>0.992299</td>
      <td>0.994645</td>
      <td>0.892188</td>
      <td>0.903815</td>
      <td>0.951434</td>
      <td>1.001248</td>
      <td>1.105228</td>
      <td>1.067897</td>
      <td>0.995979</td>
      <td>1.007514</td>
      <td>1.002216</td>
      <td>1.144957</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2010-02-24</td>
      <td>1.018720</td>
      <td>0.020933</td>
      <td>0.989414</td>
      <td>0.993331</td>
      <td>0.793902</td>
      <td>0.850671</td>
      <td>1.083166</td>
      <td>1.082661</td>
      <td>0.718941</td>
      <td>1.387077</td>
      <td>1.114644</td>
      <td>0.826805</td>
      <td>0.939438</td>
      <td>1.203928</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2010-02-25</td>
      <td>1.005786</td>
      <td>-0.012778</td>
      <td>0.986922</td>
      <td>0.991303</td>
      <td>0.740564</td>
      <td>0.799582</td>
      <td>1.041287</td>
      <td>1.051881</td>
      <td>0.621181</td>
      <td>1.412152</td>
      <td>1.185536</td>
      <td>0.930411</td>
      <td>1.020426</td>
      <td>1.256422</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
AMZN_base_value_dict
```




    {'Adj Close': 117.5199966430664,
     'TA_SMA': 122.36766688028972,
     'TA_EMA': 122.88144157847697,
     'TA_MACD': -3.4681722059076066,
     'TA_MACDSIGNAL': -4.315012307957095,
     'TA_MACDHIST': 0.8468401020494882,
     'TA_RSI': 36.1622632819538,
     'TA_MOM': -14.730003356933594,
     'KDJ_K': 44.466476440429965,
     'KDJ_D': 48.3487332418896,
     'TA_WILLR': -81.36959589165023,
     'TA_OBV': -117714400.0,
     'TA_HT_DCPERIOD': 17.064815635634197}




```python
# Prepare dataset for RNN model
AMZN_rnn_df = AMZN_features_data.loc[:,'Adj Close':'TA_HT_TRENDMODE']

# Set learning window length as 10 (trading days)
window_len = 10
n_features = AMZN_rnn_df.shape[1]
rnn_split_ratio = 0.75

# step 1: perform train-test split
raw_rnn_train_set, raw_rnn_test_set = train_test_split (AMZN_rnn_df, split_ratio = rnn_split_ratio)

# step 2: feature scaling, using min-max scaler
train_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
raw_rnn_train_set = train_scaler.fit_transform(raw_rnn_train_set)

test_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
raw_rnn_test_set = test_scaler.fit_transform(raw_rnn_test_set)

# step 3: perform "stacking" (transform a forecasting problem into a supervised problem)
rnn_train_set = train_test_prep(raw_rnn_train_set,
                                window_len = window_len,
                                output_step = 1)

rnn_test_set = train_test_prep(raw_rnn_test_set,
                                   window_len = window_len,
                                   output_step = 1)

```


```python
print(rnn_train_set.shape)
print(rnn_test_set.shape)
```

    (2037, 165)
    (672, 165)



```python
# spliting inputs from outputs
rnn_train_X, rnn_test_X, rnn_train_y, rnn_test_y = train_test_reformat(rnn_train_set.values,
                                                                       rnn_test_set.values,
                                                                       window_len = 10,
                                                                       output_step = 1)

print(rnn_test_X.shape)
```

    (672, 10, 15)



```python
print(rnn_train_X.shape)
```

    (2037, 10, 15)


### 4.2 Construct RNN with LSTM model
Here we will use Keras from Tensorflow to define and construct a RNN model, with multiple LSTM layers. Below is a note for some important model hyperparameters:
1. **number of hidden layer**: The number of hidden layers in our RNN. Here we construct two LSTM hidden layers in our RNN model. 
2. **number of neurons** (as *units*): the number of neurons in each layer of recurrent neural network. Here we set number of neurons in both hidden layers as 50.
3. **loss function** (as *loss*): the error function that we use in our training process. Here we use mean squared error function as our loss function
4. **batch size** (as *batch_size*): the number of samples used in one gradient step. 
4. **epochs** (as *epochs*): the number of times for our model to work through the entire training data set. Note that a large value of this parameter can lead to overfitting. 


```python
# Construct RNN model with LSTM hidden layers
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units = 50,
                            return_sequences = True,
                            input_shape=(rnn_train_X.shape[1],rnn_train_X.shape[2])))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.LSTM(units = 50))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam')

```


```python
# fit network, notice: epochs should not be set to large 
rnn_train_history = model.fit(rnn_train_X,
                              rnn_train_y,
                              epochs=15,
                              batch_size=60,
                              validation_data=(rnn_test_X, rnn_test_y),
                              verbose=2,
                              shuffle=False)
```

    Train on 2037 samples, validate on 672 samples
    Epoch 1/15
    2037/2037 - 13s - loss: 0.0079 - val_loss: 0.0813
    Epoch 2/15
    2037/2037 - 3s - loss: 0.0338 - val_loss: 0.0306
    Epoch 3/15
    2037/2037 - 3s - loss: 0.0226 - val_loss: 0.0022
    Epoch 4/15
    2037/2037 - 3s - loss: 0.0027 - val_loss: 0.0017
    Epoch 5/15
    2037/2037 - 3s - loss: 0.0035 - val_loss: 0.0253
    Epoch 6/15
    2037/2037 - 3s - loss: 0.0044 - val_loss: 0.0051
    Epoch 7/15
    2037/2037 - 3s - loss: 0.0019 - val_loss: 0.0016
    Epoch 8/15
    2037/2037 - 3s - loss: 0.0030 - val_loss: 0.0156
    Epoch 9/15
    2037/2037 - 3s - loss: 0.0022 - val_loss: 0.0034
    Epoch 10/15
    2037/2037 - 3s - loss: 0.0016 - val_loss: 0.0026
    Epoch 11/15
    2037/2037 - 3s - loss: 0.0015 - val_loss: 0.0037
    Epoch 12/15
    2037/2037 - 3s - loss: 0.0013 - val_loss: 0.0043
    Epoch 13/15
    2037/2037 - 3s - loss: 0.0013 - val_loss: 0.0062
    Epoch 14/15
    2037/2037 - 3s - loss: 0.0013 - val_loss: 0.0103
    Epoch 15/15
    2037/2037 - 3s - loss: 0.0014 - val_loss: 0.0019



```python
# Plot Train & Validation Loss
plt.plot(rnn_train_history.history['loss'], label='train loss')
plt.plot(rnn_train_history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()
```


    
![png](output_39_0.png)
    



```python
rnn_test_X[:, -14:]
```

### 4.3 Assessing model performance with error metrics
Here we calculate the following evaluation functions using predicted values and actual values: **Mean Squared Error**, **Mean Absolute Error** and **R Squared regression score**. The rationale of using three evaluation
functions instead of one is to reduce bias in our evaluation metric


```python
# use inputs of test set to perform prediction
rnn_y_predict = model.predict(rnn_test_X)
rnn_test_X = rnn_test_X.reshape((rnn_test_X.shape[0], window_len*n_features))
rnn_test_y = rnn_test_y.reshape((len(rnn_test_y), 1))
```


```python
print (rnn_y_predict.shape)
print (rnn_test_X.shape)
print (rnn_test_y.shape)
```

    (672, 1)
    (672, 150)
    (672, 1)



```python
# invert scaling for forecast
rnn_inv_y_predict = np.concatenate((rnn_y_predict, rnn_test_X[:, -14:]), axis=1)
rnn_inv_y_predict = test_scaler.inverse_transform(rnn_inv_y_predict)
rnn_inv_y_predict = rnn_inv_y_predict[:,0] * AMZN_base_value_dict['Adj Close']

# invert scaling for actual
rnn_inv_y = np.concatenate((rnn_test_y, rnn_test_X[:, -14:]), axis=1)
rnn_inv_y = test_scaler.inverse_transform(rnn_inv_y)
rnn_inv_y = rnn_inv_y[:,0] * AMZN_base_value_dict['Adj Close']

# calculate evaluation functions
rnn_mse_error = mean_squared_error(rnn_inv_y, rnn_inv_y_predict)
rnn_mape_error = mean_absolute_percentage_error(rnn_inv_y, rnn_inv_y_predict)
rnn_r_squared_score = r2_score(rnn_inv_y, rnn_inv_y_predict)

rnn_error_dict = {'rnn_mse_error':rnn_mse_error, 'rnn_mape_error':rnn_mape_error, 'rnn_r_squared_score':rnn_r_squared_score}
print('Test Mean_Squared_Error: %.3f' % rnn_mse_error)
print('Test Mean_Absolute_Percentage_Error: %.3f' % rnn_mape_error)
print('Test R_Squared regression score: %.3f' % rnn_r_squared_score)
```

    Test Mean_Squared_Error: 9093.240
    Test Mean_Absolute_Percentage_Error: 0.037
    Test R_Squared regression score: 0.970



```python
plt.plot(rnn_inv_y_predict, label='prediction')
plt.plot(rnn_inv_y, label='actual')
plt.title("RNN Prediction")
plt.legend()
plt.show()
```


    
![png](output_45_0.png)
    


## Section 5: Construct & Compare with Benchmark Models
To assess the performance of our RNN with LSTM model, we will introduce two extra models as our benchmark models: 
1. non-algorithmic model: Weighted Moving Average (WMA) of the adjusted close price 
2. algorithmic model: XGBoost regressor 

### 5.1 The WMA model
Note: Calculating historical price averages and using them as forecasting results does not involve a rigorous learning process but rather a process of guessing with agnostic believes upheld by many financial chartists. Given that security markets encompass a great number of latent factors (e.g., public psychology, information asymmetry, political influence…) that are incredibly intricate, it is worthwhile to introduce such a non-algorithmic model as a benchmark for reference. 


```python
def wma_price_predictor (raw_stock_data):
    
    output = raw_stock_data.copy(deep = True) 
    time_period = 30

    data_datetime_str_to_date = output['Date'].values
    for i in range (len(data_datetime_str_to_date)):
        data_datetime_str_to_date[i] = dt.datetime.strptime(data_datetime_str_to_date[i], '%Y-%m-%d').date()
        
    output['Date'] = data_datetime_str_to_date

    # USING TA-lib to calculate weighted moving average

    output['TA_WMA'] = talib.WMA(output['Adj Close'].values, time_period)
    output = output.dropna()
    output = output.drop(['Open','High','Low','Close','Volume'],1)
    return output 
```


```python
# Use WMA to predict Adj Close price
wma_test_ratio = 0.2
wma_test_index = int (AMZN_data.shape[0] * (1-wma_test_ratio))
wma_result = wma_price_predictor(AMZN_data) 
wma_real_value = wma_result['Adj Close'][wma_test_index:]
wma_prediction = wma_result['TA_WMA'][wma_test_index:]
wma_prediction_value = wma_result['TA_WMA'][wma_test_index:].values
print(wma_test_index)
```

    2209



```python
# calculate evaluation functions
wma_mse_error = mean_squared_error(wma_real_value, wma_prediction_value)
wma_mape_error = mean_absolute_percentage_error(wma_real_value, wma_prediction_value)
wma_r_squared_score = r2_score(wma_real_value, wma_prediction_value)

wma_error_dict = {'wma_mse_error':wma_mse_error, 'wma_mape_error':wma_mape_error,'wma_r_squared_score':wma_r_squared_score}

print('Test Mean_Squared_Error: %.3f' % wma_mse_error)
print('Test Mean_Absolute_Percentage_Error: %.3f' % wma_mape_error)
print('Test R_Squared regression score: %.3f' % wma_r_squared_score)
```

    Test Mean_Squared_Error: 10539.076
    Test Mean_Absolute_Percentage_Error: 0.034
    Test R_Squared regression score: 0.969



```python
plt.plot(wma_prediction, label='prediction')
plt.plot(wma_real_value, label='actual')
plt.title("WMA Prediction")
plt.legend()
plt.show()
```


    
![png](output_51_0.png)
    


### 5.2 The XGBoost model
XGBoost regressor is introduced as the second benchmark model to compare with our primary model on forecasting performance. Notice that to apply XGBoost regressor, we need to frame the stock data time-series forecasting problem into a supervised learning problem, same as what we did in the RNN model. 


```python
# Prepare dataset for XGBoost regressor
XGB_data_df = AMZN_features_data.copy()
XGB_data_df = XGB_data_df.drop(['Date'], 1)

xgb_split_ratio = 0.8

raw_rnn_train_set, raw_rnn_test_set = train_test_split (XGB_data_df, split_ratio = xgb_split_ratio)

XGB_train = train_test_prep(raw_rnn_train_set,
                                window_len = window_len,
                                output_step = 1)

XGB_test = train_test_prep(raw_rnn_test_set,
                               window_len = window_len,
                               output_step = 1)

XGB_train = np.array(XGB_train)
XGB_test = np.array(XGB_test)

```


```python
XGB_test.shape
```




    (536, 165)




```python

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-15], train[:, -15]
    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(train, test):
    predictions = list()
    train = np.asarray(train)
    test = np.asarray(test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-15], test[i, -15]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.4f, predicted=%.4f' % (testy, yhat))

    predictions = np.array(predictions)    
    return test[:, -15], predictions
```


```python
# notice: step-wise prediction of multiple hundreds data point could be time consuming, takes about 2 hours
# to check the correctness of code, use the XGB_test_check listed below

#XGB_test_check = XGB_test[:20,:]
#xgb_y_true, xgb_y_prediction = walk_forward_validation(XGB_train, XGB_test_check)

xgb_y_true, xgb_y_prediction = walk_forward_validation(XGB_train, XGB_test)

```

    /Users/jamesliang/anaconda3/lib/python3.7/site-packages/xgboost/data.py:106: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
      "because it will generate extra copies and increase " +


    >expected=14.1723, predicted=14.6771
    >expected=13.8513, predicted=14.0921
    >expected=13.9790, predicted=13.5125
    >expected=14.9378, predicted=13.4611
    >expected=14.9329, predicted=14.9453
    >expected=14.5714, predicted=14.8320
    >expected=13.9283, predicted=14.6877
    >expected=13.8799, predicted=14.4249
    >expected=13.6063, predicted=13.7440
    >expected=13.7801, predicted=13.3807
    >expected=13.5586, predicted=13.6978
    >expected=12.8684, predicted=13.5899
    >expected=12.7252, predicted=12.7295
    >expected=12.9061, predicted=12.5943
    >expected=12.7813, predicted=12.8020
    >expected=13.4558, predicted=12.9580
    >expected=13.4566, predicted=13.2700
    >expected=14.2763, predicted=13.2029
    >expected=14.2407, predicted=14.5346
    >expected=14.3820, predicted=14.3728
    >expected=15.0813, predicted=14.3076
    >expected=14.1967, predicted=15.3560
    >expected=14.4587, predicted=14.0602
    >expected=13.8626, predicted=14.3396
    >expected=13.9638, predicted=13.5266
    >expected=13.9826, predicted=14.4762
    >expected=14.1554, predicted=14.6688
    >expected=14.1115, predicted=14.3409
    >expected=13.5459, predicted=14.3610
    >expected=12.9417, predicted=13.1216
    >expected=13.2018, predicted=12.6452
    >expected=12.7219, predicted=13.2745
    >expected=12.4305, predicted=12.6521
    >expected=11.7210, predicted=12.2863
    >expected=11.4360, predicted=11.8261
    >expected=12.5162, predicted=11.7435
    >expected=12.4374, predicted=12.5660
    >expected=12.5768, predicted=12.0847
    >expected=12.7805, predicted=12.3445
    >expected=13.0967, predicted=13.2216
    >expected=12.7662, predicted=12.5322
    >expected=13.4053, predicted=12.8849
    >expected=13.8658, predicted=13.3690
    >expected=14.0962, predicted=13.3412
    >expected=14.1203, predicted=14.7416
    >expected=14.0931, predicted=14.0478
    >expected=13.9598, predicted=14.2482
    >expected=13.7611, predicted=13.4549
    >expected=14.2491, predicted=13.6950
    >expected=14.3276, predicted=14.5980
    >expected=14.4079, predicted=14.2097
    >expected=14.4333, predicted=14.3459
    >expected=13.8884, predicted=14.1329
    >expected=13.9552, predicted=13.6022
    >expected=14.0821, predicted=13.7340
    >expected=14.2152, predicted=14.6269
    >expected=13.9371, predicted=14.4205
    >expected=13.5626, predicted=13.5632
    >expected=14.2140, predicted=13.6687
    >expected=14.6250, predicted=14.5398
    >expected=13.8379, predicted=14.3981
    >expected=13.8981, predicted=13.9484
    >expected=14.1151, predicted=13.9601
    >expected=13.9573, predicted=14.4565
    >expected=13.7370, predicted=13.6532
    >expected=13.5145, predicted=13.8115
    >expected=13.5381, predicted=13.4948
    >expected=13.9381, predicted=13.7073
    >expected=13.9551, predicted=13.3650
    >expected=13.8074, predicted=13.8359
    >expected=13.6824, predicted=13.7733
    >expected=13.8494, predicted=13.7694
    >expected=13.8028, predicted=13.7548
    >expected=13.7801, predicted=13.8214
    >expected=13.8833, predicted=13.6716
    >expected=13.8955, predicted=13.7904
    >expected=13.9244, predicted=13.7078
    >expected=13.9643, predicted=13.9237
    >expected=13.9536, predicted=13.9196
    >expected=14.2251, predicted=13.9921
    >expected=14.4330, predicted=14.5867
    >expected=14.4012, predicted=14.1233
    >expected=14.2014, predicted=14.4474
    >expected=13.8355, predicted=14.2848
    >expected=13.7917, predicted=13.5934
    >expected=14.2156, predicted=13.8027
    >expected=14.2367, predicted=14.6259
    >expected=14.3874, predicted=14.4419
    >expected=14.3484, predicted=14.0950
    >expected=14.5708, predicted=14.5516
    >expected=14.8243, predicted=14.7588
    >expected=14.9919, predicted=14.6098
    >expected=15.2933, predicted=15.2591
    >expected=15.4804, predicted=15.5143
    >expected=15.0168, predicted=15.3856
    >expected=15.0975, predicted=15.0944
    >expected=15.1784, predicted=15.0322
    >expected=15.0247, predicted=15.0432
    >expected=15.0904, predicted=15.0065
    >expected=15.1527, predicted=15.0387
    >expected=15.4373, predicted=15.0064
    >expected=15.4355, predicted=15.8103
    >expected=15.4927, predicted=15.3019
    >expected=15.4770, predicted=15.5201
    >expected=15.6338, predicted=15.6266
    >expected=15.7408, predicted=15.6165
    >expected=15.6215, predicted=15.6970
    >expected=15.7193, predicted=15.6363
    >expected=15.6915, predicted=15.7548
    >expected=15.6829, predicted=15.6816
    >expected=15.6983, predicted=15.7293
    >expected=15.8530, predicted=15.7018
    >expected=15.8681, predicted=15.7138
    >expected=15.8415, predicted=15.9215
    >expected=16.0595, predicted=15.9157
    >expected=16.3697, predicted=16.2438
    >expected=16.1824, predicted=16.5225
    >expected=16.1866, predicted=16.1463
    >expected=16.5983, predicted=16.2429
    >expected=16.4945, predicted=16.5855
    >expected=16.3931, predicted=16.5158
    >expected=16.2655, predicted=16.4620
    >expected=16.1744, predicted=16.3761
    >expected=16.6989, predicted=16.1734
    >expected=16.5976, predicted=16.5297
    >expected=16.3462, predicted=16.5528
    >expected=16.3187, predicted=16.4623
    >expected=16.1664, predicted=16.0289
    >expected=16.0822, predicted=16.3107
    >expected=15.5095, predicted=16.0704
    >expected=15.6579, predicted=15.5130
    >expected=15.9220, predicted=15.4022
    >expected=16.2319, predicted=15.4623
    >expected=15.9037, predicted=16.0207
    >expected=15.8183, predicted=15.9797
    >expected=15.8060, predicted=15.8667
    >expected=15.8244, predicted=15.7956
    >expected=15.4483, predicted=15.8594
    >expected=15.5146, predicted=15.4069
    >expected=15.6265, predicted=15.5376
    >expected=15.4798, predicted=15.6304
    >expected=15.4554, predicted=15.3323
    >expected=15.1044, predicted=15.3541
    >expected=14.4034, predicted=14.9034
    >expected=14.7172, predicted=14.2591
    >expected=14.7932, predicted=15.0648
    >expected=14.9282, predicted=14.8632
    >expected=15.3508, predicted=14.9347
    >expected=15.8325, predicted=15.1638
    >expected=15.8586, predicted=15.8793
    >expected=15.7873, predicted=15.7380
    >expected=15.9147, predicted=15.8221
    >expected=15.9094, predicted=15.8386
    >expected=16.0486, predicted=15.8532
    >expected=16.1791, predicted=15.8701
    >expected=16.2423, predicted=15.9770
    >expected=16.3222, predicted=16.2949
    >expected=16.2636, predicted=16.4621
    >expected=16.2857, predicted=16.2076
    >expected=15.9826, predicted=16.4742
    >expected=16.1490, predicted=16.1470
    >expected=16.2039, predicted=16.0821
    >expected=16.1133, predicted=16.2107
    >expected=16.3563, predicted=16.1263
    >expected=16.4594, predicted=16.3573
    >expected=16.4993, predicted=16.4956
    >expected=16.5326, predicted=16.4657
    >expected=16.6127, predicted=16.6554
    >expected=16.9188, predicted=16.6104
    >expected=17.1665, predicted=17.0133
    >expected=17.0275, predicted=16.9346
    >expected=17.1120, predicted=16.6771
    >expected=17.1970, predicted=17.0493
    >expected=17.1026, predicted=16.8682
    >expected=16.9506, predicted=17.1111
    >expected=16.8303, predicted=16.9072
    >expected=16.7165, predicted=16.8538
    >expected=16.8961, predicted=16.5746
    >expected=16.9715, predicted=16.8837
    >expected=17.0253, predicted=16.9556
    >expected=16.7956, predicted=16.9862
    >expected=16.5338, predicted=16.5921
    >expected=16.2734, predicted=16.5675
    >expected=16.1550, predicted=16.2682
    >expected=15.8848, predicted=16.3239
    >expected=15.7873, predicted=15.5968
    >expected=15.5143, predicted=15.6100
    >expected=15.0198, predicted=15.4916
    >expected=15.2130, predicted=14.6059
    >expected=15.2604, predicted=15.0643
    >expected=15.5964, predicted=15.7117
    >expected=15.3810, predicted=15.5676
    >expected=15.1882, predicted=15.5266
    >expected=15.5237, predicted=14.7054
    >expected=15.0014, predicted=15.4719
    >expected=15.1133, predicted=14.9443
    >expected=15.2533, predicted=14.9690
    >expected=15.4537, predicted=15.3352
    >expected=15.3283, predicted=15.7504
    >expected=15.5168, predicted=15.5563
    >expected=15.3562, predicted=15.4689
    >expected=14.8878, predicted=15.3617
    >expected=15.0517, predicted=14.8894
    >expected=14.9917, predicted=14.6578
    >expected=15.0123, predicted=14.9782
    >expected=15.2008, predicted=15.0811
    >expected=15.1148, predicted=14.9132
    >expected=15.2301, predicted=14.9897
    >expected=15.3218, predicted=15.0638
    >expected=15.6630, predicted=15.6038
    >expected=15.6017, predicted=15.7193
    >expected=15.5833, predicted=15.6865
    >expected=15.4914, predicted=15.5624
    >expected=15.5122, predicted=15.6217
    >expected=15.6871, predicted=15.6496
    >expected=15.6513, predicted=15.5561
    >expected=15.3833, predicted=15.6516
    >expected=15.5084, predicted=15.5791
    >expected=15.4651, predicted=15.4734
    >expected=15.4995, predicted=15.5477
    >expected=15.2668, predicted=15.4804
    >expected=15.1915, predicted=15.3549
    >expected=14.8197, predicted=15.2688
    >expected=15.0471, predicted=14.9133
    >expected=14.8046, predicted=15.1281
    >expected=14.6822, predicted=15.0728
    >expected=14.7712, predicted=15.1573
    >expected=14.7690, predicted=14.6159
    >expected=14.5782, predicted=14.6942
    >expected=14.6734, predicted=14.6544
    >expected=14.8030, predicted=14.6197
    >expected=14.7435, predicted=14.7411
    >expected=14.5125, predicted=14.7006
    >expected=14.6527, predicted=14.6545
    >expected=14.6380, predicted=14.6890
    >expected=14.7372, predicted=14.5190
    >expected=14.7756, predicted=14.5759
    >expected=15.0390, predicted=14.6519
    >expected=15.1245, predicted=15.1208
    >expected=15.2100, predicted=15.1988
    >expected=14.9550, predicted=15.1985
    >expected=15.1945, predicted=15.0727
    >expected=15.0249, predicted=15.0791
    >expected=14.9946, predicted=15.0528
    >expected=15.1530, predicted=15.1525
    >expected=14.9875, predicted=14.8800
    >expected=15.1215, predicted=15.0348
    >expected=14.9992, predicted=15.0141
    >expected=15.1463, predicted=15.0482
    >expected=15.1179, predicted=15.0324
    >expected=15.2437, predicted=14.9345
    >expected=15.3562, predicted=15.1329
    >expected=15.3311, predicted=15.3994
    >expected=15.2805, predicted=15.5289
    >expected=15.2161, predicted=15.4017
    >expected=15.1964, predicted=14.9495
    >expected=15.0753, predicted=15.0935
    >expected=15.1293, predicted=15.0575
    >expected=14.9175, predicted=14.9898
    >expected=14.9302, predicted=14.9796
    >expected=14.8017, predicted=15.0822
    >expected=14.9126, predicted=14.3698
    >expected=14.9148, predicted=14.9956
    >expected=14.8530, predicted=14.9816
    >expected=14.7610, predicted=14.9490
    >expected=14.8547, predicted=14.4556
    >expected=15.0939, predicted=15.1001
    >expected=15.2905, predicted=15.1510
    >expected=15.4740, predicted=15.4991
    >expected=15.3234, predicted=15.5627
    >expected=15.1600, predicted=15.4443
    >expected=15.0609, predicted=15.0005
    >expected=14.9820, predicted=15.1399
    >expected=14.8101, predicted=15.1977
    >expected=14.9047, predicted=14.9261
    >expected=14.8869, predicted=14.9340
    >expected=14.7993, predicted=14.8544
    >expected=14.8802, predicted=14.8672
    >expected=14.9790, predicted=14.8705
    >expected=14.9842, predicted=14.8676
    >expected=15.0545, predicted=14.8513
    >expected=15.2371, predicted=15.0397
    >expected=15.1807, predicted=15.5364
    >expected=15.2509, predicted=15.1606
    >expected=15.2017, predicted=15.5207
    >expected=15.2570, predicted=15.0398
    >expected=15.2247, predicted=15.5011
    >expected=15.9017, predicted=14.9288
    >expected=15.9105, predicted=15.7157
    >expected=15.7155, predicted=15.8786
    >expected=15.7236, predicted=15.7576
    >expected=16.1505, predicted=15.7792
    >expected=15.9545, predicted=16.3031
    >expected=16.1920, predicted=16.0941
    >expected=16.2258, predicted=16.3838
    >expected=16.0991, predicted=16.2645
    >expected=16.1764, predicted=16.2081
    >expected=16.0242, predicted=16.3315
    >expected=16.0934, predicted=16.0510
    >expected=15.9074, predicted=16.1157
    >expected=15.8443, predicted=15.7103
    >expected=15.9797, predicted=15.7630
    >expected=15.8673, predicted=16.1054
    >expected=16.0994, predicted=15.6724
    >expected=16.0608, predicted=16.1376
    >expected=16.0362, predicted=16.0441
    >expected=15.8410, predicted=15.9569
    >expected=15.5577, predicted=15.6561
    >expected=15.7697, predicted=15.6078
    >expected=15.8101, predicted=15.5892
    >expected=15.9180, predicted=15.7677
    >expected=17.0926, predicted=15.3980
    >expected=17.0541, predicted=16.9072
    >expected=17.4410, predicted=16.7022
    >expected=17.3576, predicted=17.1439
    >expected=17.4458, predicted=17.1639
    >expected=17.6930, predicted=17.2201
    >expected=18.1578, predicted=17.0187
    >expected=18.3016, predicted=17.5677
    >expected=18.3799, predicted=18.0844
    >expected=18.2937, predicted=17.7304
    >expected=18.1660, predicted=18.0733
    >expected=18.3430, predicted=18.3511
    >expected=18.4668, predicted=18.2102
    >expected=18.3211, predicted=18.2232
    >expected=17.8350, predicted=18.2450
    >expected=17.0974, predicted=17.9030
    >expected=16.7864, predicted=16.3887
    >expected=16.8447, predicted=15.9051
    >expected=16.0339, predicted=17.0005
    >expected=16.0292, predicted=15.8459
    >expected=16.6265, predicted=15.8579
    >expected=16.2440, predicted=16.3786
    >expected=16.8127, predicted=16.1414
    >expected=16.3719, predicted=16.3634
    >expected=16.1767, predicted=16.0876
    >expected=15.3217, predicted=16.0697
    >expected=16.0979, predicted=15.1475
    >expected=15.4940, predicted=16.3661
    >expected=14.2666, predicted=15.5415
    >expected=15.1889, predicted=14.9379
    >expected=14.3733, predicted=15.1459
    >expected=15.3833, predicted=14.8761
    >expected=15.5718, predicted=15.1075
    >expected=16.0052, predicted=15.2456
    >expected=15.7087, predicted=16.1654
    >expected=16.1915, predicted=15.7317
    >expected=16.5087, predicted=16.4240
    >expected=16.0470, predicted=16.4424
    >expected=16.6396, predicted=16.2799
    >expected=16.1683, predicted=16.2584
    >expected=16.7116, predicted=16.4532
    >expected=16.5905, predicted=16.7819
    >expected=16.2330, predicted=17.1013
    >expected=16.3277, predicted=16.5264
    >expected=16.2235, predicted=16.1906
    >expected=16.9979, predicted=16.3742
    >expected=17.1171, predicted=16.6376
    >expected=17.3843, predicted=17.3063
    >expected=17.3822, predicted=17.4239
    >expected=18.4553, predicted=17.5036
    >expected=19.4292, predicted=18.0698
    >expected=19.6365, predicted=18.4656
    >expected=20.4917, predicted=19.0153
    >expected=20.2093, predicted=19.8372
    >expected=20.3677, predicted=20.2158
    >expected=19.8104, predicted=19.9673
    >expected=20.1114, predicted=19.8646
    >expected=20.4174, predicted=20.0240
    >expected=20.5090, predicted=19.8688
    >expected=20.2178, predicted=19.6148
    >expected=19.6909, predicted=19.1760
    >expected=20.1898, predicted=19.2030
    >expected=21.0517, predicted=18.8157
    >expected=19.4523, predicted=20.0639
    >expected=19.7072, predicted=19.5277
    >expected=19.7226, predicted=19.8449
    >expected=20.0073, predicted=20.0704
    >expected=20.1464, predicted=20.1970
    >expected=20.2486, predicted=19.7636
    >expected=20.4986, predicted=20.0812
    >expected=20.0557, predicted=19.9722
    >expected=20.1491, predicted=19.7191
    >expected=20.3272, predicted=20.2369
    >expected=20.5053, predicted=20.2048
    >expected=20.6455, predicted=20.4318
    >expected=20.8418, predicted=20.5722
    >expected=21.2554, predicted=20.7581
    >expected=20.8198, predicted=20.8872
    >expected=20.7359, predicted=20.4013
    >expected=20.6081, predicted=20.6980
    >expected=20.5105, predicted=20.7848
    >expected=20.4314, predicted=20.5693
    >expected=20.7826, predicted=20.3473
    >expected=21.0265, predicted=20.6261
    >expected=21.0382, predicted=20.6625
    >expected=21.0892, predicted=20.8487
    >expected=20.9377, predicted=20.9794
    >expected=21.1283, predicted=20.9352
    >expected=21.4777, predicted=20.9513
    >expected=22.1312, predicted=20.8949
    >expected=22.5277, predicted=22.0019
    >expected=21.7662, predicted=22.1023
    >expected=21.6561, predicted=21.8392
    >expected=21.8914, predicted=21.4714
    >expected=22.2538, predicted=21.8031
    >expected=22.4726, predicted=22.0747
    >expected=22.5832, predicted=21.9622
    >expected=22.7622, predicted=22.3186
    >expected=23.0924, predicted=22.4011
    >expected=23.5229, predicted=22.6252
    >expected=23.2675, predicted=23.1712
    >expected=23.4392, predicted=23.0926
    >expected=22.9141, predicted=23.2142
    >expected=22.8079, predicted=23.0046
    >expected=23.4753, predicted=22.8520
    >expected=24.4954, predicted=23.1287
    >expected=24.5941, predicted=24.0081
    >expected=26.0129, predicted=24.2645
    >expected=25.5286, predicted=25.4349
    >expected=26.2178, predicted=25.4948
    >expected=27.0816, predicted=25.4137
    >expected=27.2294, predicted=26.5209
    >expected=26.4125, predicted=26.9810
    >expected=26.2423, predicted=26.4703
    >expected=25.6030, predicted=26.5216
    >expected=25.5267, predicted=26.1640
    >expected=25.2040, predicted=25.7191
    >expected=27.2025, predicted=25.2639
    >expected=26.7043, predicted=25.9079
    >expected=26.3777, predicted=26.4946
    >expected=25.4131, predicted=26.6531
    >expected=25.6034, predicted=26.1122
    >expected=25.9974, predicted=25.4218
    >expected=25.5304, predicted=25.4646
    >expected=25.8129, predicted=25.8401
    >expected=25.9690, predicted=25.7308
    >expected=26.9289, predicted=25.7852
    >expected=26.4797, predicted=26.1453
    >expected=26.7089, predicted=26.1850
    >expected=27.2722, predicted=26.6390
    >expected=27.4421, predicted=26.4717
    >expected=26.9525, predicted=27.2442
    >expected=26.7883, predicted=26.9779
    >expected=26.2140, predicted=26.7093
    >expected=26.9081, predicted=26.2783
    >expected=26.8977, predicted=26.6144
    >expected=26.7871, predicted=26.7999
    >expected=27.0797, predicted=26.8070
    >expected=28.1866, predicted=26.8645
    >expected=27.7440, predicted=27.5006
    >expected=28.0579, predicted=27.0918
    >expected=27.9503, predicted=27.7513
    >expected=28.1438, predicted=27.8894
    >expected=28.4759, predicted=27.6739
    >expected=29.2874, predicted=27.6934
    >expected=28.9312, predicted=28.4038
    >expected=28.9466, predicted=28.4928
    >expected=29.3649, predicted=28.5253
    >expected=29.7747, predicted=29.2870
    >expected=30.0498, predicted=29.0992
    >expected=28.6590, predicted=29.5555
    >expected=28.0345, predicted=28.9873
    >expected=26.8026, predicted=28.2081
    >expected=27.8132, predicted=26.8263
    >expected=27.0176, predicted=27.2243
    >expected=26.5165, predicted=27.5547
    >expected=26.4038, predicted=26.6474
    >expected=26.8561, predicted=26.0910
    >expected=26.1921, predicted=26.8116
    >expected=25.6019, predicted=25.8904
    >expected=25.1439, predicted=25.9127
    >expected=25.1912, predicted=25.1665
    >expected=26.6252, predicted=25.3528
    >expected=25.5264, predicted=26.4795
    >expected=25.6960, predicted=25.6065
    >expected=26.3370, predicted=25.6636
    >expected=27.0086, predicted=26.3253
    >expected=26.7604, predicted=26.9056
    >expected=26.7931, predicted=26.9886
    >expected=27.4103, predicted=26.9026
    >expected=26.5912, predicted=26.7328
    >expected=27.2226, predicted=27.0455
    >expected=26.3781, predicted=27.0213
    >expected=27.1927, predicted=26.6001
    >expected=27.1490, predicted=26.5037
    >expected=27.9667, predicted=27.1111
    >expected=29.2965, predicted=27.3490
    >expected=29.3025, predicted=28.4482
    >expected=28.6224, predicted=28.1615
    >expected=28.4092, predicted=28.0716
    >expected=27.8481, predicted=28.3723
    >expected=27.2908, predicted=27.5811
    >expected=27.3741, predicted=27.5042
    >expected=27.1013, predicted=27.5996
    >expected=27.0286, predicted=26.9462
    >expected=27.2668, predicted=26.8863
    >expected=27.2893, predicted=27.0093
    >expected=27.9640, predicted=27.1123
    >expected=26.9127, predicted=28.2407
    >expected=27.3231, predicted=27.5134
    >expected=25.8352, predicted=26.8197
    >expected=25.5657, predicted=26.6220
    >expected=25.9395, predicted=25.7188
    >expected=27.5796, predicted=26.1517
    >expected=28.2675, predicted=27.3111
    >expected=28.1771, predicted=28.0594
    >expected=26.7507, predicted=28.1929
    >expected=25.8256, predicted=26.7898
    >expected=26.6966, predicted=26.2007
    >expected=26.4660, predicted=26.4184
    >expected=26.6236, predicted=27.1236
    >expected=26.6428, predicted=27.0683
    >expected=26.6819, predicted=27.1465
    >expected=26.4249, predicted=26.8915
    >expected=26.5233, predicted=26.3328
    >expected=26.3734, predicted=26.6924
    >expected=26.3648, predicted=26.3314
    >expected=26.5322, predicted=27.0628
    >expected=27.1024, predicted=27.1763
    >expected=27.1898, predicted=27.3907
    >expected=26.9575, predicted=27.3337
    >expected=27.4003, predicted=27.2051
    >expected=27.2594, predicted=27.4182
    >expected=27.1165, predicted=27.4797
    >expected=26.9110, predicted=27.4215
    >expected=26.8720, predicted=27.0933
    >expected=27.0362, predicted=27.0476
    >expected=26.4142, predicted=27.4351
    >expected=26.3912, predicted=26.6041
    >expected=26.5182, predicted=26.7335
    >expected=26.8633, predicted=26.4627
    >expected=26.9326, predicted=27.2274
    >expected=27.5779, predicted=26.7435
    >expected=27.5364, predicted=27.7037
    >expected=27.2434, predicted=27.7072



```python
# inverse scaling with base value
xgb_inv_y_true = xgb_y_true * AMZN_base_value_dict['Adj Close']
xgb_inv_y_prediction = np.array(xgb_y_prediction) * AMZN_base_value_dict['Adj Close']

# estimate prediction error
xgb_mse_error = mean_squared_error(xgb_inv_y_true, xgb_inv_y_prediction)
xgb_mape_error = mean_absolute_percentage_error(xgb_inv_y_true, xgb_inv_y_prediction)
xgb_r_squared_score = r2_score(xgb_inv_y_true, xgb_inv_y_prediction)

xgb_error_dict = {'xgb_mse_error':xgb_mse_error, 'xgb_mape_error':xgb_mape_error, 'xgb_r_squared_score':xgb_r_squared_score}

print('Test Mean_Squared_Error: %.3f' % xgb_error_dict['xgb_mse_error'])
print('Test Mean_Absolute_Percentage_Error: %.3f' % xgb_error_dict['xgb_mape_error'])
print('Test R_Squared regression score: %.3f' % xgb_error_dict['xgb_r_squared_score'])

```

    Test Mean_Squared_Error: 3440.967
    Test Mean_Absolute_Percentage_Error: 0.019
    Test R_Squared regression score: 0.990



```python
print (xgb_inv_y_prediction)
```


```python
# plot expected vs preducted
plt.plot(xgb_inv_y_true, label='Actual')
plt.plot(xgb_inv_y_prediction, label='Prediction')
plt.title("XGB Prediction")
plt.legend()
plt.show()
```


    
![png](output_59_0.png)
    


### 5.3 Compare primary model with benchmark models


```python
rnn_error_vec = list(rnn_error_dict.values())
wma_error_vec = list(wma_error_dict.values())
xgb_error_vec = list(xgb_error_dict.values())

error_table = pd.DataFrame(np.array([rnn_error_vec, wma_error_vec, xgb_error_vec]),
                            columns = ['mean squared error', 'mean absolute percentage error', 'r2 score'],
                            index = ['RNN','WMA','XGB'])
```


```python
error_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean squared error</th>
      <th>mean absolute percentage error</th>
      <th>r2 score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RNN</th>
      <td>9093.240080</td>
      <td>0.036755</td>
      <td>0.970160</td>
    </tr>
    <tr>
      <th>WMA</th>
      <td>10539.075718</td>
      <td>0.033934</td>
      <td>0.969481</td>
    </tr>
    <tr>
      <th>XGB</th>
      <td>3440.967248</td>
      <td>0.018862</td>
      <td>0.990000</td>
    </tr>
  </tbody>
</table>
</div>


