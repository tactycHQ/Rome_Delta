#Author: Anubhav Srivastava
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

_TESTSTART='11/10/2015'
_TESTEND='11/30/2017'

def main():
    ticker=_TICKER
    d=DataLoader(ticker)
    d.loadData()
    d.prepData()
    window_size=_WINDOWSIZE
    window_shift=_WINDOW_SHIFT
    start_index=d.getIndex(_TESTSTART)
    end_index=d.getIndex(_TESTEND)
    x_test,y_test,dates_test=createInputs(d.features,
                                           d.targets,
                                           d.dates,
                                           window_size,
                                           window_shift,
                                           start_index,
                                           end_index)


    model=SequenceModel()
    model.modelLoad("Data/"+ticker+'.h5',"Data/"+ticker+'_history.json')

    start_price = np.array(d.prices[start_index:end_index]).reshape(-1,1)

    y_pred=model.predict_model(x_test)

    y_pred = d.denormalize(y_pred, d.targets_mean, d.targets_std)
    y_pred = y_pred + 1
    y_pred_price=[]
    for i in range (0,len(y_pred)):
        cume_change = np.cumprod(y_pred[i,:,:]).reshape(-1,1)
        y_pred_price.append(cume_change*start_price[i])
    y_pred_price=np.array(y_pred_price)

    y_dates = d.dates[start_index+1:end_index]

    y_actuals = d.prices[start_index+1:end_index]

    plotTestPerformance(y_pred_price,y_actuals,y_dates,model.history_dict,d.targets_std,window_size=window_size)


def plotPerformance(model,history,targets_std):
    loss = history['loss']
    val_loss = history['val_loss']
    print('Training loss (Denormalized)', loss[-1] * targets_std)
    print('Validation loss (Denormalized)', val_loss[-1]*targets_std)

    fig, axs =plt.subplots()
    axs.plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
    axs.plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
    axs.set_title('Training and validation loss')
    axs.legend()
    fig.autofmt_xdate()
    plt.show()

def plotTestPerformance(y_pred,y_actuals,y_dates,history,targets_std,window_size):

    loss = history['loss']
    val_loss = history['val_loss']
    print('Training loss (Denormalized)', loss[-1] * targets_std)
    print('Validation loss (Denormalized)', val_loss[-1]*targets_std)

    fig, axs =plt.subplots(2,1)
    axs[0].plot(range(1, len(loss) + 1), loss, 'bo', label='Training loss')
    axs[0].plot(range(1, len(loss) + 1), val_loss, 'b', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()

    y_dates = y_dates.flatten()
    # Convert y_dates timed to datetime format
    for i in range (0,len(y_dates)):
        y_dates[i] = dt.datetime.strptime(y_dates[i], '%m/%d/%Y')


    axs[1].plot(y_dates, y_actuals, label='Actual Prices')
    for p in range(0, len(y_pred), window_size):
        y_pred_timed = y_pred[p, :, :].flatten()
        y_pred_timed = np.lib.pad(y_pred_timed,
                                  (p, max(0,len(y_dates) - p - window_size)),
                                  'constant',
                                  constant_values=np.NAN)
        y_pred_timed=y_pred_timed[:y_dates.shape[0]]
        axs[1].plot(y_dates, y_pred_timed, label='Predicted Prices')

    axs[1].set_title('Actuals vs Predicted')
    axs[1].legend(loc=(1.04,0))
    axs[1].fmt_xdata = mdates.DateFormatter('%m-%d-%Y')
    fig.autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    main()


