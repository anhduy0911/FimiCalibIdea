import matplotlib.pyplot as plt
import pandas as pd

def plot_data():
    import matplotlib.dates as mdates

    aqm1 = pd.read_csv("Data/fimi/fimi_13.csv", header=0)
    aqm3 = pd.read_csv("Data/fimi/fimi_12.csv", header=0)
    print(aqm1.describe())
    print(aqm3.describe())

    aqm1_part = pd.DataFrame()
    aqm1_part['PM2_5'] = aqm1['PM2_5'][:17012]
    aqm1_part['temp'] = aqm1['temperature'][:17012]
    aqm1_part['humidity'] = aqm1['humidity'][:17012]
    # print(aqm1_part.describe())
    # aqm3_part = pd.DataFrame()
    aqm1_part['PM2_5_cal'] = aqm3['PM2_5']
    aqm1_part['temp_cal']= aqm3['temperature']
    aqm1_part['humidity_cal'] = aqm3['humidity']
    aqm1_part.to_csv('data/fimi/12_13.csv', header=True, index=False)
    # print(aqm3_part.describe())
    # aqm3_part.to_csv('Data/aqmes3_part.csv', header=True, index=False)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    ax1.plot(range(len(aqm1['PM2_5'])),aqm1['PM2_5'])
    ax1.plot(range(len(aqm3['PM2_5'])),aqm3['PM2_5'])
    ax1.set(xlabel='Date', ylabel='PM2.5')
    ax2.plot(range(len(aqm1['PM2_5'])),aqm1['temperature'])
    ax2.plot(range(len(aqm3['PM2_5'])),aqm3['temperature'])
    ax2.set(xlabel='Date', ylabel='temp')
    ax3.plot(range(len(aqm1['PM2_5'])),aqm1['humidity'])
    ax3.plot(range(len(aqm3['PM2_5'])),aqm3['humidity'])
    ax3.set(xlabel='Date', ylabel='humidity')
    
    plt.show()


if __name__ == '__main__':
    plot_data()
    # import numpy as np
    # aqm = pd.read_csv("Data/aqmes1_part.csv", header=0)
    # index = aqm['PM2_5'].index[aqm['PM2_5'].apply(np.isnan)]
    # index_calib = aqm['PM2_5_cal'].index[aqm['PM2_5_cal'].apply(np.isnan)]
    # print(f'index for pm: {index}, pm_calib: {index_calib}')