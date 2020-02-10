import matplotlib.pyplot as plt
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


# size = get_index_size('cpu_usage', 'data')
# print(size)

def predict(index):
    data = csv_reader(index_name=index, size=10000)
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
    df.loc[(df["Value"] >= '100000') & (df["Value"] < '1000000'), "Value_comb"] = '13'
    df.loc[(df["Value"] >= '1000000') & (df["Value"] < '10000000'), "Value_comb"] = '14'
    df.loc[(df["Value"] >= '10000000') & (df["Value"] < '100000000'), "Value_comb"] = '15'
    df.loc[(df["Value"] >= '100000000') & (df["Value"] < '1000000000'), "Value_comb"] = '16'
    df.loc[(df["Value"] >= '1000000000') & pd.notnull(df["Value"]), "Value_comb"] = '17'

    df["Value_comb"] = df["Value_comb"].astype(float)
    # data["Time"] = pd.to_datetime(data["Time"])

    df["Num_Date_Time"] = df["Date_Time"].str.replace(":", ".").str[0:5]
    # df["Num_Date_Time"] = df["Num_Date_Time"].str[3:8]
    df["Num_Date_Time"] = df["Num_Date_Time"].astype(float)

    new_data = df[["Num_Date_Time", "Value_comb"]].copy()

    indexnewData = new_data.set_index(["Num_Date_Time"])

    rollmean = indexnewData.rolling(window=13).mean()
    rollstd = indexnewData.rolling(window=13).std()

    new_data.to_csv('export_dataframe.csv')

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
    model = ARResults.load('pickle_model.pkl')
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

    x_predict = []
    y_predict = []
    x_test = []
    y_test = []

    for data in predictions:
        point = data.shape
        if len(point) == 1:
            x_predict.append(point[0])
            y_predict.append(None)
        else:
            x_predict.append(point[0])
            y_predict.append(point[1])

    for data in test:
        point = data.shape
        if len(point) == 1:
            x_test.append(point[0])
            y_test.append(None)
        else:
            x_test.append(point[0])
            y_test.append(point[1])

    # result = ((x_predict, y_predict), (x_test, y_test))
    result = (predictions, test)
    return result


if __name__ == "__main__":
    t = predict('cpu_usage')
    # print(type(t[0]))
    # print(type(t[0][0]))
    # print(t[0][0].shape)
    # print(t[0])
    print(len(t[0]))
    # print(t[1])
    plt.plot(t[0])
    # plt.plot(t[1])
    plt.show()

    # predictions = ([1, 1, 2, 2, 2, 3, 3], [10, 15, 20, 25, 27, 30, 35])
    # data = dict()
    # size = len(set(predictions[0]))
    #
    # x_predict = [None]*size
    # y_predict = [[None]*size]
    # old_x = predictions[0][0]
    # x_predict[0] = old_x
    # y_predict[0][0] = predictions[1][0]
    # count = 0
    # index = 1
    # for i in range(1, len(predictions[0])):
    #     x, y = predictions[0][i], predictions[1][i]
    #     if old_x == x:
    #         count += 1
    #         if count == len(y_predict):
    #             y_predict.append([None]*size)
    #         y_predict[count][index-1] = y
    #     else:
    #         count = 0
    #         x_predict[index] = x
    #         y_predict[count][index] = y
    #         old_x = x
    #         index += 1
    #
    # print(x_predict, y_predict)
