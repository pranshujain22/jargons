import pandas as pd
import numpy as np
from pandas import read_csv
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import ARResults
import numpy
import pickle
from dashboard.Elastic import csv_reader, get_index_size
from pandasticsearch import Select

lines = get_index_size('cpu_usage', 'data')
print(lines)

data = csv_reader(index_name='cpu_usage', size=10000)
# print(json.dumps(data, indent=4))
df = Select.from_dict(data).to_pandas()

df["Time"] = df["Time"].str.replace("T", " ").str[0:19]

df_col_time = df["Time"].str.split(' ', expand=True)
df["Date"] = df_col_time[0]
df["Date_Time"] = df_col_time[1]

df["Value_comb"] = '0'

df.loc[pd.isnull(df["Value"]), "Value_comb"] = '0'
df.loc[(df["Value"] > '0') & (df["Value"] < '0.000001'), "Value_comb"] = '1'
df.loc[(df["Value"] >= '0.000001') & (df["Value"] < '0.00001'), "Value_comb"] = '2'
df.loc[(df["Value"] >= '0.00001') & (df["Value"] < '0.0001'), "Value_comb"] = '3'
df.loc[(df["Value"] >= '0.0001') & (df["Value"] < '0.001'), "Value_comb"] = '4'
df.loc[(df["Value"] >= '0.001') & (df["Value"] < '0.01'), "Value_comb"] = '5'
df.loc[(df["Value"] >= '0.01') & (df["Value"] < '0.1'), "Value_comb"] = '6'
df.loc[(df["Value"] >= '0.1') & (df["Value"] < '1'), "Value_comb"] = '7'
df.loc[(df["Value"] >= '1') & (df["Value"] < '10'), "Value_comb"] = '8'
df.loc[(df["Value"] >= '10') & (df["Value"] < '100'), "Value_comb"] = '9'
df.loc[(df["Value"] >= '100') & (df["Value"] < '1000'), "Value_comb"] = '10'
df.loc[(df["Value"] >= '1000') & (df["Value"] < '10000'), "Value_comb"] = '11'
df.loc[(df["Value"] >= '10000') & (df["Value"] < '100000'), "Value_comb"] = '12'
df.loc[(df["Value"] >= '100000') & pd.notnull(df["Value"]), "Value_comb"] = '13'

df["Value_comb"] = df["Value_comb"].astype(float)
# data["Time"] = pd.to_datetime(data["Time"])

df["Num_Date_Time"] = df["Date_Time"].str.replace(":", ".").str
df["Num_Date_Time"] = df["Num_Date_Time"].astype(float)

new_data = df[["Num_Date_Time", "Value_comb"]].copy()

indexnewData = new_data.set_index(["Num_Date_Time"])

rollmean = indexnewData.rolling(window=13).mean()
rollstd = indexnewData.rolling(window=13).std()

new_data.to_csv(r'export_dataframe.csv')

series = read_csv('export_dataframe.csv', header=0, index_col=0)

series.set_index(['Num_Date_Time'], inplace=True)


# create a difference transform of the dataset
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return numpy.array(diff)


# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


# split dataset
X = difference(series.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]

# train autoregression
window_size = 6

model = AR(train)
model_fit = model.fit(maxlag=window_size, disp=False)

# save model to file
# model_fit.save('ar_model.pkl')
# save model using pickle
pickle.dump(model_fit, open('pickle_model.pkl', 'wb'))
# save the differenced dataset
numpy.save('ar_data.npy', X)
# save the last observation
numpy.save('ar_obs.npy', [series.values[-1]])
# save coefficients
coef = model_fit.params
numpy.save('man_model.npy', coef)
# save lag
lag = X[-window_size:]
numpy.save('man_data.npy', lag)

window = model_fit.k_ar
coef = model_fit.params

# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(test)):
    yhat = predict(coef, history)
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# load the AR model from file
model = ARResults.load('ar_model.pkl')
# print(loaded.params)
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
print(last_ob)
coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)

# make prediction
# prediction = predict(coef, lag)
prediction = predict(coef, lag)
# transform prediction
yhat = prediction + last_ob[0]
print('Prediction: %f' % yhat)

