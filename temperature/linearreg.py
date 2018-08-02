import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %X')
data = pd.read_csv('dataA1_0515-19_temperature.csv', parse_dates=['timestamp'], date_parser=dateparse)
# print(data.head)
# print(data.index)
# print(data.dtypes)
# print(data['timestamp'])
# print(data['temperature'])


def timestamp_to_int(tst):
    tst = tst.to_pydatetime()
    tst = tst.hour * 3600 + tst.minute * 60 + tst.second
    return tst


data['timestamp'] = data.timestamp.apply(timestamp_to_int)

# print(type(data))
# print(type(data['timestamp']))
# print(type(data['timestamp'][0]))
# print(type(data['timestamp'].values))

array = data['timestamp'].values
array = array.reshape(-1,1)

# linear regression
from sklearn import linear_model
regr = linear_model.LinearRegression()

regr.fit(array, data['temperature'])

'''
DRAW
'''


plt.scatter(data['timestamp'], data['temperature'], color='blue')
plt.plot(data['timestamp'], regr.predict(array), color='red', linewidth=4)
plt.xlabel('time')
plt.ylabel('temperature')
plt.show()

'''
EVALUATE
'''

RMSE = np.sqrt(((data['temperature'] - regr.predict(array)) ** 2).mean())
from sklearn.metrics import r2_score
r2 = abs(r2_score(data['temperature'], regr.predict(array)))
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(data['temperature'], regr.predict(array))
print(f'RMSE:{RMSE} R2: {r2} MAPE:{mape}')