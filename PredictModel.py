# Author: Anubhav Srivastava
#License: MIT License

from DataLoader import DataLoader
from SequenceModel import SequenceModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import datetime as dt
from BuildModel import _WINDOW_SHIFT,_WINDOWSIZE,_TICKER
from BuildModel import createInputs
import sys
import pandas as pd
import pandas_market_calendars as mcal

_PREDSTART='11/20/2018'

def main():

    ticker = _TICKER
    window_size=_WINDOWSIZE
    window_shift=_WINDOW_SHIFT

    d = DataLoader(ticker)
    d.loadData()
    d.prepData()

    x_history=d.features[-window_size:,:]
    start_price=d.prices[-1]
    print("Start price: ",start_price)
    x_history=x_history.reshape(1,x_history.shape[0],x_history.shape[1])

    model = SequenceModel()
    model.modelLoad("Data/" + ticker + '.h5', "Data/" + ticker + '_history.json')

    y_pred = model.predict_model(x_history)
    y_pred_delta=d.denormalize(y_pred,d.targets_mean,d.targets_std)
    y_pred_delta = y_pred_delta.flatten().reshape(-1, 1)
    print(y_pred_delta.shape)

    y_pred_price=PricefromDelta(y_pred_delta,start_price)

    plot_dates=dates_axis(window_size,d)
    plotPredictions(y_pred_price,d,plot_dates)


def PricefromDelta(y_pred_delta,start_price):
    cume_change = np.cumprod(1+y_pred_delta)
    y_pred_prices=start_price*cume_change
    y_pred_prices = np.array(y_pred_prices).reshape(-1,1)
    print(y_pred_prices.shape)
    return y_pred_prices

def plotPredictions(y_pred_price,d,plot_dates):
    proj_width=len(y_pred_price)
    hist_width=2*proj_width

    recent_prices = d.prices[-hist_width:,:]
    recent_prices = np.lib.pad(recent_prices, (0,proj_width), 'constant', constant_values=np.NAN)
    recent_prices = recent_prices[:,0]
    print(recent_prices)
    print(plot_dates)

    y_pred_price=np.lib.pad(y_pred_price,(hist_width,0),'constant',constant_values=np.NAN)
    y_pred_price=y_pred_price[:,-1]

    fig, axs = plt.subplots()
    axs.plot(plot_dates,y_pred_price)
    axs.plot(plot_dates,recent_prices,label='Historical Prices')
    axs.set_title('Predictions')
    axs.fmt_xdata = mdates.DateFormatter('%m-%d-%Y')
    fig.autofmt_xdate()
    plt.show()

def dates_axis(window_size,d):

    plot_start=d.dates[d.getIndex(_PREDSTART)+2-window_size*2]
    width=window_size*3
    nyse=mcal.get_calendar('NYSE')
    all_dates_raw = nyse.valid_days(start_date=plot_start[0], end_date='2030-12-31')
    plot_dates=all_dates_raw[:width]

    return plot_dates


if __name__ == '__main__':
    main()


