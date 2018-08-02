
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as st
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# get data


def dateparse(dates): return pd.datetime.strptime(dates, '%Y-%m-%d %X')


data = pd.read_csv('dataA1_0515-19_temperature.csv',
                   parse_dates=['timestamp'], index_col='timestamp', date_parser=dateparse)
# print(data.head())
# print(data.index)
# print(data['timestamp'].shape) # (2264,)
# print(data['temperature'].shape) # (2264,)
# print(type(data['timestamp'][0])) # <class 'pandas._libs.tslibs.timestamps.Timestamp'>
# print(type(data['temperature'][0])) # <class 'numpy.float64'>

# threshold = 0.5
# data['pandas'] = data['temperature'].rolling(3).median()
# difference = np.abs(data['temperature'] - data['pandas'])
# outlier_idx = difference > threshold
# # print(f'{sum(outlier_idx)} outliers') # 155 outliers
# data.drop(data[outlier_idx].index, inplace=True)


def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]


# smooth

ts = data['temperature']
ts_log = np.log(ts)

# for i in range(2,15):
#     rolling_avg = ts_log.rolling(i).mean()
#     rolling_avg.dropna(inplace=True)
#     print(i,test_stationarity(rolling_avg))
# 5 is the best

rolling_avg = ts_log.rolling(5).mean()
rolling_avg.dropna(inplace=True)

ts_diff = rolling_avg.diff(1)
ts_diff.dropna(inplace=True)
# print(test_stationarity(ts_diff))


# find best model


best_p = 1
best_q = 1
best_RMSE = 10000

for p in range(1, 7):
    for q in range(1, 7):
        try:
            model = ARIMA(ts_diff, order=(p, 1, q))
            result_arma = model.fit(disp=-1, method='css')
            predict_ts = result_arma.predict()
            # 一阶差分还原
            diff_shift_ts = rolling_avg.shift(1)
            diff_recover = predict_ts.add(diff_shift_ts)
            # 移动平均还原
            rol_sum = ts_log.rolling(window=5).sum()
            rol_recover = diff_recover*6 - rol_sum.shift(1)
            # 对数还原
            log_recover = np.exp(rol_recover)
            log_recover.dropna(inplace=True)
            cur = np.sqrt(sum((log_recover-ts)**2)/ts.size)
            if cur < best_RMSE:
                best_p = p
                best_q = q
                best_RMSE = cur
        except:
            pass

# plot

model = ARIMA(ts_diff, order=(best_p, 1, best_q))
result_arma = model.fit(disp=-1, method='css')
predict_ts = result_arma.predict()

diff_shift_ts = rolling_avg.shift(1)
diff_recover = predict_ts.add(diff_shift_ts)

rol_sum = ts_log.rolling(window=5).sum()
rol_recover = diff_recover*6 - rol_sum.shift(1)

log_recover = np.exp(rol_recover)
log_recover.dropna(inplace=True)

cur = np.sqrt(sum((log_recover-ts)**2)/ts.size)
ts = ts[log_recover.index]
sns.set()
plt.figure(facecolor='white')
log_recover.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.ylabel('temperature')
plt.legend(loc='best')
plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover-ts)**2)/ts.size))
plt.show()
