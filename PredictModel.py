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

#Test Class
#TESTSTART defines which date to start prediction from
#TESTEND is last date of prediction

_PREDSTART='11/10/2018'

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

    y_pred_price=PricefromDelta(y_pred_delta,start_price)

    plotPredictions(y_pred_price,d)


def PricefromDelta(y_pred_delta,start_price):
    cume_change = np.cumprod(1+y_pred_delta)
    y_pred_prices=start_price*cume_change
    y_pred_prices = np.array(y_pred_prices).reshape(-1,1)

    return y_pred_prices

def plotPredictions(y_pred_price,d):
    hist_width=2*len(y_pred_price)

    y_pred_price=np.lib.pad(y_pred_price,(hist_width,0),'constant',constant_values=np.NAN)
    fig, axs = plt.subplots()
    axs.plot(y_pred_price)
    axs.plot(d.prices[-hist_width:,:],label='Historical Prices')
    axs.set_title('Predictions')
    axs.legend()
    plt.show()


if __name__ == '__main__':
    main()


