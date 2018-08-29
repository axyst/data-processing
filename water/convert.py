import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as st
import seaborn as sns
import warnings
import datetime
warnings.filterwarnings('ignore')


def dateparse(dates): return pd.datetime.strptime(dates, '%Y/%m/%d %H:%M')


data = pd.read_csv('dataset_1.csv',
                   parse_dates=['record_time'],  date_parser=dateparse)

data.drop(['id'], axis=1, inplace=True)
# print(data.head())

counts = {}
beginning = {}
ending = {}
new_col = []
total_hour = 0
strings = data['record_time'].astype(str)
idx = 0
all_time_list = []

for item in data['record_time']:

    s_time = strings[idx]
    c_time = item
    c_year = item.year
    c_month = item.month
    c_day = item.day
    c_hour = item.hour
    time_list = (c_year, c_month, c_day, c_hour)

    if time_list not in counts:
        counts[time_list] = 1
        beginning[time_list] = c_time
        ending[time_list] = c_time
        total_hour = total_hour + 1
        all_time_list.append(time_list)
    else:
        counts[time_list] = counts[time_list] + 1
        ending[time_list] = c_time

    idx = idx + 1

result = {'time': [], 'begin_time': [], 'end_time': [],
          'received': [], 'expected': [], 'packet_loss': []}
results = pd.DataFrame(data=result)

for timelist in all_time_list:
    # print(timelist, counts[timelist], beginning[timelist], ending[timelist])
    real = counts[timelist]
    expected = (ending[timelist] - beginning[timelist]).total_seconds()
    expected = (expected % 3600) // 60
    expected = np.ceil(expected / 2 + 1)
    ct = str(timelist[0])+"/"+str(timelist[1])+"/" + \
        str(timelist[2])+" hour: "+str(timelist[3])
    cb = beginning[timelist]
    ce = ending[timelist]
    cp = 1-real/expected
    to_add = [ct, cb, ce, real, expected, cp]
    results.loc[len(results)] = to_add


# print(results.head())
results.to_csv('output.csv', index=False)
