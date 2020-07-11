from Features import *
import fxcmpy
import pandas as pd
import datetime as dt
import pickle
import math
from datetime import timedelta
#------------------------------------------------------------------------------
#defult data
Sympol = 'EUR/USD'
Period = 'H1'
start_traning = dt.datetime(2015, 1, 1) 
stop_traning = dt.datetime(2018, 8, 1)
stop_testing = dt.datetime(2019, 7, 8) 
#------------------------------------------------------------------------------
#Get Data From Market

#add more bars than 10000
dateTimeDifference = start_traning - stop_testing
dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
dateTimeDifferenceInHours = abs(dateTimeDifferenceInHours)
i = dateTimeDifferenceInHours / 9000
i = math.ceil(i)


TOKEN = "fea322e8370d263a0556673b882dfdf8d9065699"
        
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

print('Data is ready to process...')
print(' ')

#------------------------------------------------------------------------------
#Process Data

momentumKey = [3,4,5,8,9,10] 
stochasticKey = [3,4,5,8,9,10] 
williamsKey = [6,7,8,9,10] 
procKey = [12,13,14,15] 
wadlKey = [15] 
adoscKey = [2,3,4,5] 
macdKey = [15,30] 
cciKey = [15] 
bollingerKey = [15] 
paverageKey = [2] 
slopeKey = [3,4,5,10,20,30] 
fourierKey = [10,20,30] 
sineKey = [5,6] 
#----------------------
keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
            ,paverageKey,slopeKey,fourierKey,sineKey] 

momentumDict = momentum(prices,momentumKey) 
print('1')
stochasticDict = stochastic(prices,stochasticKey) 
print('2')
williamsDict = williams(prices,williamsKey)
print('3')
procDict = proc(prices,procKey) 
print('4')
wadlDict = wadl(prices,wadlKey)
print('5') 
adoscDict = adosc(prices,adoscKey)
print('6')
macdDict = macd(prices,macdKey) 
print('7')
cciDict = cci(prices,cciKey) 
print('8')
bollingerDict = bollinger(prices,bollingerKey,2) 
print('9')
paverageDict = pavarage(prices,paverageKey) 
print('10')
slopeDict = slopes(prices,slopeKey) 
print('11')
fourierDict = fourier(prices,fourierKey) 
print('12')
sineDict = sine(prices,sineKey) 
print('13')
#----------------------
# Create list of dictionaries 

dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
            ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
            ,cciDict.cci,bollingerDict.bands,paverageDict.avs
            ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs] 

# list of column name on csv

colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
            'cci','bollinger','paverage','slope','fourier','sine',]

masterFrame = pd.DataFrame(index = prices.index) 
for i in range(0,len(dictlist)): 
    if colFeat[i] == 'macd':
        colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1]) 
        masterFrame[colID] = dictlist[i] 
    else: 
        for j in keylist[i]: 
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + str(k)
                masterFrame[colID] = dictlist[i][j][k]
                    
threshold = round(0.7*len(masterFrame)) 
masterFrame[['open','high','low','close','volume']] = prices[['open','high','low','close','volume']]

masterFrameCleaned = masterFrame.copy() 
masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)

#------------------------------------------------------------------------------
#for traning
s = 0 
masterFrameCleaned['Date']=pd.to_datetime(masterFrameCleaned.index)
for i in range(0,len(masterFrameCleaned)):
    if masterFrameCleaned.Date.iloc[i] >= stop_traning:
        s = i
        break


train = masterFrameCleaned.iloc[:s]

prices.to_csv('data/prices.csv')

train.to_csv('data/calculated.csv')

#for backtesting
test = masterFrameCleaned.iloc[s:]
test.to_csv('data/calculated1.csv')
#------------------------------------------------------------------------------    

print('Complete procrss the features...')
print(' ')
#------------------------------------------------------------------------------