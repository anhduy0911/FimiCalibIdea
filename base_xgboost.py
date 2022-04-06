from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# input selection
def input_selection(data,i,seq_length,option = 0):
  if option == 0: # only pm2.5 
    return data[i:(i+seq_length), 0]
  elif option == 1: # pm2.5 + temp
    return data[i:(i+seq_length), 0:2]
  elif option == 2: # pm2.5 + humid
    return data[i:(i+seq_length), 0:3:2]
  else: # any other option is select all !
    return data[i:(i+seq_length),0:3]

def get_index_feat(option = 0):
  if option == 0: # only pm2.5 
    return [0]
  elif option == 1: # pm2.5 + temp
    return [0,1]
  elif option == 2: # pm2.5 + humid
    return [0,2]
  else: # any other option is select all !
    return [0,1,2]

#Data Processing
def scaling_window(training_data, testing_data, seq_length_in,seq_length_out,option = 0):
    x = []
    y = []
    z = []

    for i in range(len(training_data)-seq_length_in-1):
        _x = input_selection(training_data,i,seq_length_in,option)
        _y = testing_data[(i+seq_length_in-seq_length_out+1):(i+seq_length_in+1), 0]
        _z = training_data[(i+seq_length_in-seq_length_out+1):(i+seq_length_in+1), 0]
        x.append(_x)
        y.append(_y)
        z.append(_z)
    x, y, z = np.array(x,dtype=float),np.array(y, dtype=float), np.array(z,dtype=float)
    # print(f'y shape: {y.shape}')
    # print(f'x shape: {x.shape}')

    if len(x.shape) < 3:
      x = x.reshape((x.shape[0], x.shape[1]))
    if len(y.shape) < 3:
      y = y.reshape((y.shape[0], y.shape[1]))
    if len(z.shape) < 3:
      z = z.reshape((z.shape[0], z.shape[1]))

    x = x.reshape(x.shape[0], -1)
    # print(f'x shape: {x.shape}')
    return x, y, z

def get_data(data,seq_length_in,seq_length_out,option=0):
  tmp_arr = np.vstack((data[0, :], data[4, :])).T
#   print(tmp_arr.shape)
  sc = MinMaxScaler()
  tmp_arr = sc.fit_transform(tmp_arr)
#   print(tmp_arr[:10])
  data_x = tmp_arr[:, 0].reshape(-1, 1)
  data_y = tmp_arr[:, 1].reshape(-1, 1)
#   print(data_x.shape)
#   print(data_y.shape)
  # seq_length = 7
  x, y, z = scaling_window(data_x, data_y, seq_length_in,seq_length_out,option)

  train_size = int(len(y) * 0.6)
  test_size = len(y) - train_size

  dataX = x
  dataY = y
  before_calibration = z

  trainX = np.array(x[:train_size, :])
  trainY = np.array(y[:train_size])

  testX = np.array(x[-test_size:, :])
  testY = np.array(y[-test_size:])

#   print(trainX.shape)
#   print(trainY.shape)
#   print(dataX.shape)
  return dataX,dataY,before_calibration,trainX,trainY,testX,testY, sc

def xgb_exp(data,seq_length_in,seq_length_out,option):
  dataX,dataY,before_calibration,trainX,trainY,testX,testY, sc = get_data(data,seq_length_in,seq_length_out,option)
  train_size = int(len(dataY) * 0.6)
  test_size = len(dataY) - train_size

  xgb = XGBRegressor(booster='gbtree', verbosity=0, 
                   nthread=-1, eta=0.4,
                   max_depth=4, grow_policy='depthwise', 
                   objective='reg:squarederror', seed=99)
  multip = MultiOutputRegressor(xgb)
  multip.fit(trainX, trainY)
  data_predict = multip.predict(dataX)
  real_data = dataY

  original_data = before_calibration
#   print(original_data[:10])
  if data_predict.shape[0] == 1:
    data_predict = np.squeeze(data_predict, axis=0)
  if original_data.shape[0] == 1:
    original_data = np.squeeze(original_data, axis=0)
  if real_data.shape[0] == 1:
    real_data = np.squeeze(real_data, axis=0)
#   print(data_predict.shape)
  tmp_arr = np.hstack((data_predict, real_data))
#   print(tmp_arr.shape)
  tmp_arr = sc.inverse_transform(tmp_arr)
  data_predict = tmp_arr[:, 0].reshape(-1,1)
  real_data = tmp_arr[:, 1].reshape(-1,1)
#   print(data_predict.shape)
  tmp_arr = np.hstack((original_data, real_data))
  tmp_arr = sc.inverse_transform(tmp_arr)
  original_data = tmp_arr[:, 0].reshape(-1,1)

  list_feature = ["PM2_5"]
  index_feature = get_index_feat(0)
  # print(index_feature)
  if len(data_predict.shape) < 2:
    data_predict = np.reshape(data_predict, (data_predict.shape[0], 1))
    original_data = np.reshape(original_data, (original_data.shape[0], 1))
    real_data = np.reshape(real_data, (real_data.shape[0], 1))
  for index in index_feature:
    i = index_feature.index(index)
    # mae1 = mean_absolute_error(real_data[-test_size:, i], original_data[-test_size:, i])
    # mse1 = mean_squared_error(real_data[-test_size:, i], original_data[-test_size:, i])
    # mape1 = mean_absolute_percentage_error(real_data[-test_size:, i], original_data[-test_size:, i])
    error = real_data[-test_size:, i] - original_data[-test_size:, i]
    rmse1 = np.sqrt(np.square(error).mean())
    mae1 = np.abs(error).mean()
    mape1 = np.abs(error / real_data[-test_size:, i]).mean() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(8)
    fig.set_figwidth(22)
    fig.suptitle(list_feature[index])
    # ax1.axvline(x=train_size, c='g', linestyle='--')
    ax1.plot(real_data[-test_size:, i], label='Real')
    ax1.plot(original_data[-test_size:, i], label='Original')
    ax1.set_title("Before calibration")
    ax1.legend()

    print(f'Before calib: MAE: {str(mae1)}, RMSE: {str(rmse1)}, MAPE: {str(mape1)}')
    mae3 = mean_absolute_error(real_data[-test_size:, i], data_predict[-test_size:, i])
    # mse2 = mean_squared_error(real_data[-test_size:, i], data_predict[-test_size:, i])
    # mape2 = mean_absolute_percentage_error(real_data[-test_size:, i], data_predict[-test_size:, i])
    error = real_data[-test_size:, i] - data_predict[-test_size:, i]
    rmse2 = np.sqrt(np.square(error).mean())
    mae2 = np.abs(error).mean()
    mape2 = np.abs(error / real_data[-test_size:, i]).mean() * 100
    print(f'After calib: MAE: {str(mae2)}, RMSE: {str(rmse2)}, MAPE: {str(mape2)}')

    # ax2.axvline(x=train_size, c='g', linestyle='--')
    ax2.plot(real_data[-test_size:, i], label='Real')
    ax2.plot(data_predict[-test_size:, i], label='Prediction')
    ax2.set_title("After calibration")
    ax2.legend()
    fig.savefig('img/xgboost.png')

if __name__ == '__main__':
    aqm = pd.read_csv('Data/fimi/envitus_fimi14.csv', header=0).values[:,1:].T
    print(aqm)
    xgb_exp(aqm,32,1,0)