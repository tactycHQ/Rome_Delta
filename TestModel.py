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
_TESTEND='11/20/2018'

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

    print("Mean: ", d.targets_mean)
    print("STD: ", d.targets_std)
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
    # print("y_test 1st value: ", d.denormalize(y_test[0,0,0],d.targets_mean,d.targets_std))
    print("dates_test shape: ", dates_test.shape)

    model=SequenceModel()
    model.modelLoad("Data/"+ticker+'.h5',"Data/"+ticker+'_history.json')
    y_pred=model.predict_model(x_test)
    y_pred_price,y_actuals,y_dates=createPlotData(start_index,end_index,y_pred,d)

    # checkModel(d,x_test, y_actuals, y_dates, start_index)

    plotTestPerformance(y_pred_price,y_actuals,y_dates,model.history_dict,d.targets_std,window_size=window_size)


def createPlotData(start_index, end_index, y_pred,d):
    start_price = np.array(d.prices[start_index:end_index]).reshape(-1,1)
    y_pred = d.denormalize(y_pred, d.targets_mean, d.targets_std)
    y_pred = y_pred + 1
    y_pred_price = []
    for i in range(0, len(y_pred)):
        cume_change = np.cumprod(y_pred[i, :, :]).reshape(-1, 1)
        y_pred_price.append(cume_change * start_price[i])
    y_pred_price = np.array(y_pred_price)
    y_dates = d.dates[start_index + 1:end_index]
    y_actuals = d.prices[start_index + 1:end_index]

    return y_pred_price,y_actuals,y_dates

def checkModel(data,x_test,y_actuals,y_dates, start_index):
    print("Date of 1st y_actuals", y_dates[0])
    print("Value of 1st y_actuals", y_actuals[0])
    print("Value of 1st y_test", y_actuals[0])
    print("Value of 1st x_test", data.denormalize(x_test[0,0,:],data.targets_mean,data.targets_std))
    print("Value of 1st prices", data.prices[start_index+1])


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


