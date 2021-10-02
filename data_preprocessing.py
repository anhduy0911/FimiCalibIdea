import datetime
import matplotlib.pyplot as plt
import pandas as pd

def plot_data():
    aqm1 = pd.read_csv("Data/aqmesh1.csv", header=0)[26:1458]
    aqm2 = pd.read_csv("Data/aqmesh2.csv", header=0)[25:1650]
    aqm3 = pd.read_csv("Data/aqmesh3.csv", header=0)[26:1659]
    fimi2 = pd.read_csv("Data/fimi2.csv", header=0)[13169:17572:3]
    print(aqm2.describe())
    print(aqm3.describe())

    aqm2['datetime'] = pd.to_datetime(aqm2['datetime'], format='%Y-%m-%d %H:%M:%S') 
    aqm3['datetime'] = pd.to_datetime(aqm3['datetime'], format='%Y-%m-%d %H:%M:%S')
    fimi2['datetime'] = pd.to_datetime(fimi2['datetime'], format='%m/%d/%Y %H:%M').dt.round('5min')

    # print(fimi2['datetime'].head())
    # print(aqm2['datetime'].head())
    # print(aqm3['datetime'].head())

    merged = pd.merge(aqm3, fimi2, how='outer', on='datetime')
    clean_dat = pd.DataFrame()
    clean_dat['datetime'] = merged['datetime']
    clean_dat['PM2_5'] = merged['PM2_5_y']
    clean_dat['temp'] = merged['temp_y']
    clean_dat['humidity'] = merged['humidity_y']
    clean_dat['PM2_5_cal'] = merged['PM2_5_x']
    clean_dat['temp_cal'] = merged['temp_x']
    clean_dat['humidity_cal'] = merged['humidity_x']    

    clean_dat = clean_dat.dropna(axis=0)
    clean_dat.to_csv('Data/aqmesh_fimi.csv', index=None)
    print(clean_dat.head(10))
    

    #plot

    fig, (ax1) = plt.subplots(1,1, sharex=True)
    ax1.plot(merged['datetime'],merged['PM2_5_x'], 'b')
    ax1.plot(merged['datetime'],merged['PM2_5_y'], 'r')
    # ax1.set(xlabel='Date', ylabel='PM2.5')
    # # ax2.plot(range(len(aqm2['PM2_5'])),aqm2['temp'], 'b-')
    # # ax2.plot(range(len(aqm3['PM2_5'])),aqm3['temp'], 'g-')
    # # ax2.plot(range(len(fimi2['PM2_5'])),fimi2['temp'], 'r-')
    # # ax2.set(xlabel='Date', ylabel='temp')
    # # ax3.plot(range(len(aqm2['PM2_5'])),aqm2['humidity'], 'b-')
    # # ax3.plot(range(len(aqm3['PM2_5'])),aqm3['humidity'], 'g-')
    # # ax3.plot(range(len(fimi2['PM2_5'])),fimi2['humidity'], 'r-')
    # # ax3.set(xlabel='Date', ylabel='humidity')
    
    plt.gcf().autofmt_xdate()
    plt.show()

def plot_data_grim():
    grim3 = pd.read_csv("Data/grim3.csv", header=0)
    grim7 = pd.read_csv("Data/grim7.csv", header=0)
    fimi2 = pd.read_csv("Data/fimi2.csv", header=0)

    grim3['datetime'] = pd.to_datetime(grim3['datetime'], format='%Y-%m-%d %H:%M:%S')
    grim7['datetime'] = pd.to_datetime(grim7['datetime'], format='%Y-%m-%d %H:%M:%S').dt.round('5min')
    fimi2['datetime'] = pd.to_datetime(fimi2['datetime'], format='%m/%d/%Y %H:%M').dt.round('5min')
    print(grim7['datetime'].head())
    print(fimi2['datetime'].head())

    merged = pd.merge(grim7, fimi2, how='outer', on='datetime')
    clean_dat = pd.DataFrame()
    clean_dat['datetime'] = merged['datetime']
    clean_dat['PM2_5'] = merged['PM2_5_y']
    clean_dat['temp'] = merged['temp']
    clean_dat['humidity'] = merged['humidity']
    clean_dat['PM2_5_cal'] = merged['PM2_5_x']
    # clean_dat['temp_cal'] = merged['temp_x']
    # clean_dat['humidity_cal'] = merged['humidity_x']    

    clean_dat = clean_dat.dropna(axis=0)
    clean_dat.to_csv('Data/grim_fimi.csv', index=None)
    print(clean_dat.head(10))
    #plot

    fig, (ax1) = plt.subplots(1,1, sharex=True)
    ax1.plot(clean_dat['datetime'],clean_dat['PM2_5'], 'g-')
    ax1.plot(clean_dat['datetime'],clean_dat['PM2_5_cal'], 'r-')
    # ax1.set(xlabel='Date', ylabel='PM2.5')
    # ax2.plot(range(len(aqm2['PM2_5'])),aqm2['temp'], 'b-')
    # ax2.plot(range(len(aqm3['PM2_5'])),aqm3['temp'], 'g-')
    # ax2.plot(range(len(fimi2['PM2_5'])),fimi2['temp'], 'r-')
    # ax2.set(xlabel='Date', ylabel='temp')
    # ax3.plot(range(len(aqm2['PM2_5'])),aqm2['humidity'], 'b-')
    # ax3.plot(range(len(aqm3['PM2_5'])),aqm3['humidity'], 'g-')
    # ax3.plot(range(len(fimi2['PM2_5'])),fimi2['humidity'], 'r-')
    # ax3.set(xlabel='Date', ylabel='humidity')
    
    plt.gcf().autofmt_xdate()
    plt.show()

def new_dataset():
    envitus = pd.read_csv("Data/fimi/envitus.csv", header=0)
    envitus['datetime'] = pd.to_datetime(envitus['datetime'], format='%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=25200) # 7 hours
    envitus['datetime'] = pd.to_datetime(envitus['datetime'], format='%Y-%m-%d %H:%M:%S').dt.round('1min')
    # envitus.to_csv('Data/fimi/envitus.csv', index=None)

    fimi1 = pd.read_csv("Data/fimi/fimi14.csv", header=0)
    fimi1.index = pd.DatetimeIndex(fimi1['datetime'])

    mean_fimi1 = pd.DataFrame()
    mean_fimi1['PM2_5'] = fimi1['PM2_5'].resample('1Min').mean().round(2)
    mean_fimi1['PM10_0'] = fimi1['PM10_0'].resample('1Min').mean().round(2)
    mean_fimi1['temp'] = fimi1['temp'].resample('1Min').mean().round(2)
    mean_fimi1['humidity'] = fimi1['humidity'].resample('1Min').mean().round(2)
    # mean_fimi1.to_csv('Data/fimi/mean_fimi1.csv', index=True)

    merged = pd.merge(mean_fimi1, envitus, how='outer', on='datetime')
    clean_dat = pd.DataFrame()
    clean_dat['datetime'] = merged['datetime']
    clean_dat['PM2_5'] = merged['PM2_5_x']
    clean_dat['PM10_0'] = merged['PM10_0_x']
    clean_dat['temp'] = merged['temp_x']
    clean_dat['humidity'] = merged['humidity_x']
    clean_dat['PM2_5_cal'] = merged['PM2_5_y']
    clean_dat['temp_cal'] = merged['temp_y']
    clean_dat['humidity_cal'] = merged['humidity_y'] 
    clean_dat['PM10_0_cal'] = merged['PM10_0_y']

    clean_dat = clean_dat.dropna(axis=0)
    print(clean_dat.head())
    clean_dat.to_csv('Data/fimi/envitus_fimi14.csv', index=None)

def plot_data_new():
    merged = pd.read_csv("Data/fimi/envitus_fimi14.csv", header=0)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    ax1.plot(merged['datetime'],merged['PM2_5'], 'b')
    ax1.plot(merged['datetime'],merged['PM2_5_cal'], 'r')
    ax1.set(xlabel='Date', ylabel='PM2.5')
    ax2.plot(merged['datetime'],merged['temp'], 'b')
    ax2.plot(merged['datetime'],merged['temp_cal'], 'r')
    ax2.set(xlabel='Date', ylabel='temp')
    ax3.plot(merged['datetime'],merged['humidity'], 'b')
    ax3.plot(merged['datetime'],merged['humidity_cal'], 'r')
    ax3.set(xlabel='Date', ylabel='humidity')

    plt.gcf().autofmt_xdate()
    plt.show()

if __name__ == '__main__':
    # plot_data()
    # new_dataset()
    plot_data_new()
    # plot_data_grim()
    # import numpy as np
    # aqm = pd.read_csv("Data/aqmes1_part.csv", header=0)
    # index = aqm['PM2_5'].index[aqm['PM2_5'].apply(np.isnan)]
    # index_calib = aqm['PM2_5_cal'].index[aqm['PM2_5_cal'].apply(np.isnan)]
    # print(f'index for pm: {index}, pm_calib: {index_calib}')