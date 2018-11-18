#Author: Anubhav Srivastava
#License: MIT License

from iexfinance import get_historical_data
import pandas as pd
import numpy as np
import datetime as dt

#Helper Class to load and prepare data from csv file in the /data director

class DataLoader():

    def __init__(self,ticker):
        self.ticker=ticker
        self.stock_data=None
        self.fname = "data/"+self.ticker + '.csv'
        self.features=None
        self.prices=None
        self.dates=None
        self.features_mean=None
        self.features_std=None
        self.prices_mean = None
        self.prices_std = None

#getData functional is optional. To download data from IEX API directly if CSV not available. IEX data is only 5 years though while Nasdaq data can go back 10 years
    def getData(self):
        start = dt.datetime(2013, 2, 9)
        end = dt.datetime(2017, 5, 9)
        self.stock_data = get_historical_data(self.ticker, start=start, end=end, output_format='pandas')
        self.stock_data.to_csv(self.fname)
        print("Saved to", self.fname)

        return self.stock_data

    def loadData(self):
        self.stock_data = pd.read_csv(self.fname)
        self.stock_data = self.stock_data.iloc[::-1]
        print("------Loaded Raw Data-------")

    def prepData(self):

        self.targets=[]

        self.prices = self.stock_data['close'].values.reshape(-1, 1)
        for i in range(1,len(self.prices)):
            self.targets.append((self.prices[i]-self.prices[i-1])/self.prices[i-1])
        self.targets=np.array(self.targets)

        self.volume = self.stock_data['volume'].values.reshape(-1, 1)
        self.volume_diff = np.diff(self.volume,axis=0)
        self.volume_diff = self.volume_diff.reshape(-1, 1)

        self.dates = self.stock_data['date'].values.reshape(-1, 1)
        self.dates=self.dates[1:,:]
        self.features=np.concatenate((self.targets,self.volume_diff),axis=1)

        self.features, self.features_mean, self.featured_std = self.normalize(self.features)
        self.targets, self.targets_mean, self.targets_std = self.normalize(self.targets)

        print("Feature Shape is ", self.features.shape)
        print("Target Shape is ", self.targets.shape)
        print("Dates Shape is ", self.dates.shape)

#Standard Scaler normalization
    def normalize(self,denorm):
        mean = denorm.mean(axis=0)
        std = denorm.std(axis=0)
        norm = (denorm - mean) / std

        return norm, mean, std

    def denormalize(self,norm,mean,std):
        denorm = norm * std + mean
        return denorm

    def getIndex(self, date):
        i=np.where(self.dates==date)[0][0]
        return i






