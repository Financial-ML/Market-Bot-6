import fxcmpy
import pandas as pd
import datetime as dt
import pickle
import math
from datetime import timedelta
#------------------------------------------------------------------------------
#defult data
Sympol = 'AUD/USD'
Period = 'H1'
start_traning = dt.datetime(2016, 1, 1) 
stop_traning = dt.datetime(2019, 10, 1) 
stop_testing = dt.datetime(2020, 7, 1)  
#------------------------------------------------------------------------------
#Get Data From Market

#add more bars than 10000
dateTimeDifference = start_traning - stop_testing
dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
dateTimeDifferenceInHours = abs(dateTimeDifferenceInHours)
i = dateTimeDifferenceInHours / 9000
i = math.ceil(i)


TOKEN = "eb73b905098d6d601a15892fabb03965f596240a"
        
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
        
if con.is_connected() == True:
    print("Data retrieved...")
    print(' ')
    for x in range(i-1):
        stop_traning1 = start_traning + timedelta(days=375)
        df = con.get_candles(Sympol, period=Period, start=start_traning, stop=stop_traning1)
        df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow'])
        df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
        df = df[['open','high','low','close','volume']]
        df = df[~df.index.duplicated()]
        if x==0:
            prices = df.copy()
        else:
            prices = pd.concat([prices,df])
        start_traning = start_traning + timedelta(days=375)

    df = con.get_candles(Sympol, period=Period, start=start_traning, stop=stop_testing)
    df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow']) 
    df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
    df = df[['open','high','low','close','volume']]
    df = df[~df.index.duplicated()]
    if i==1:
        prices = df.copy() 
    else:
        prices = pd.concat([prices,df])
else:
    print('No connection with fxcm')

prices = prices.drop_duplicates(keep=False)
con.close()

s = 0 
prices['Date']=pd.to_datetime(prices.index)
for i in range(0,len(prices)):
    if prices.Date.iloc[i] >= stop_traning:
        s = i
        break


prices1 = prices.iloc[:s]
prices2 = prices.iloc[s:]

prices1.to_csv('data/prices1.csv')
prices2.to_csv('data/prices2.csv')