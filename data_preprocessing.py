import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

    clean_dat = clean_dat.dropna(axis=0)
    clean_dat.to_csv('Data/grim_fimi.csv', index=None)
    print(clean_dat.head(10))
    #plot

    fig, (ax1) = plt.subplots(1,1, sharex=True)
    ax1.plot(clean_dat['datetime'],clean_dat['PM2_5'], 'g-')
    ax1.plot(clean_dat['datetime'],clean_dat['PM2_5_cal'], 'r-')
    
    plt.gcf().autofmt_xdate()
    plt.show()

def new_dataset():
    envitus = pd.read_csv("Data/fimi/envitus.csv", header=0)
    envitus['datetime'] = pd.to_datetime(envitus['datetime'], format='%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=25200) # 7 hours
    envitus['datetime'] = pd.to_datetime(envitus['datetime'], format='%Y-%m-%d %H:%M:%S').dt.round('1min')
    
    envitus.index = pd.DatetimeIndex(envitus['datetime'])
    
    envitus_mean = pd.DataFrame()
    envitus_mean['PM2_5'] = envitus['PM2_5'].resample('30Min').mean().round(2)
    envitus_mean['PM10_0'] = envitus['PM10_0'].resample('30Min').mean().round(2)
    envitus_mean['temp'] = envitus['temp'].resample('30Min').mean().round(2)
    envitus_mean['humidity'] = envitus['humidity'].resample('30Min').mean().round(2)
    # envitus.to_csv('Data/fimi/envitus.csv', index=None)
    print(envitus_mean.head())
    fimi1 = pd.read_csv("Data/fimi/fimi20.csv", header=0)
    fimi1.index = pd.DatetimeIndex(fimi1['datetime'])

    mean_fimi1 = pd.DataFrame()
    mean_fimi1['PM2_5'] = fimi1['PM2_5'].resample('30Min').mean().round(2)
    mean_fimi1['PM10_0'] = fimi1['PM10_0'].resample('30Min').mean().round(2)
    mean_fimi1['temp'] = fimi1['temp'].resample('30Min').mean().round(2)
    mean_fimi1['humidity'] = fimi1['humidity'].resample('30Min').mean().round(2)
    # mean_fimi1.to_csv('Data/fimi/mean_fimi1.csv', index=True)
    print(mean_fimi1.head())

    merged = pd.merge(mean_fimi1, envitus_mean, how='inner', on='datetime')
    print(merged.head())
    clean_dat = pd.DataFrame()
    clean_dat.index = merged.index
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
    clean_dat.to_csv('Data/fimi/envitus_fimi14.csv', index=True)

def resample_dataset():
    fimi1 = pd.read_csv("Data/fimi_new/envitus_fimi14.csv", header=0)
    fimi1.index = pd.DatetimeIndex(fimi1['datetime'])

    mean_fimi1 = pd.DataFrame()
    mean_fimi1.index = fimi1.index
    mean_fimi1['PM2_5'] = fimi1['PM2_5'].resample('30Min').mean().round(2)
    mean_fimi1['PM10_0'] = fimi1['PM10_0'].resample('30Min').mean().round(2)
    mean_fimi1['temp'] = fimi1['temp'].resample('30Min').mean().round(2)
    mean_fimi1['humidity'] = fimi1['humidity'].resample('30Min').mean().round(2)
    mean_fimi1['PM2_5_cal'] = fimi1['PM2_5_cal'].resample('30Min').mean().round(2)
    mean_fimi1['PM10_0_cal'] = fimi1['PM10_0_cal'].resample('30Min').mean().round(2)
    mean_fimi1['temp_cal'] = fimi1['temp_cal'].resample('30Min').mean().round(2)
    mean_fimi1['humidity_cal'] = fimi1['humidity_cal'].resample('30Min').mean().round(2)
    clean_dat = mean_fimi1.dropna(axis=0)
    print(clean_dat.head())
    clean_dat.to_csv('Data/fimi/envitus_fimi14_mean.csv', index=True)

def resample_data(file_name, envitus=False):
    path = "Data/fimi_new/"
    path_save = "Data/fimi_resample/"
    fimi = pd.read_csv(path + file_name, header=0)
    if envitus:
        fimi['datetime'] = pd.to_datetime(fimi['datetime'], format='%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=25200)
    fimi['datetime'] = pd.to_datetime(fimi['datetime'], format='%Y-%m-%d %H:%M:%S').dt.round('1min')
    fimi.index = pd.DatetimeIndex(fimi['datetime'])

    mean_fimi = pd.DataFrame()
    mean_fimi.index = fimi.index

    mean_fimi['PM2_5'] = fimi['PM2_5'].resample('1Min').mean().round(2)
    mean_fimi['PM10'] = fimi['PM10'].resample('1Min').mean().round(2)
    mean_fimi['temp'] = fimi['temp'].resample('1Min').mean().round(2)
    mean_fimi['humidity'] = fimi['humidity'].resample('1Min').mean().round(2)
    # mean_fimi['PM1_0'] = fimi['PM1_0'].resample('1Min').mean().round(2)
    mean_fimi['CO'] = fimi['CO'].resample('1Min').mean().round(2)
    mean_fimi['NO2'] = fimi['NO2'].resample('1Min').mean().round(2)
    mean_fimi['SO2'] = fimi['SO2'].resample('1Min').mean().round(2)
    
    clean_dat = mean_fimi.groupby(level=0).mean()
    clean_dat.index = pd.DatetimeIndex(clean_dat.index)
    print(clean_dat.head())
    clean_dat.to_csv(path_save + file_name, index=True)

def plot_data_new():
    merged = pd.read_csv("Data/fimi_resample/envitus_fimi_overlapped.csv", header=0)
    atts = ['PM2_5', 'PM10', 'temp', 'humidity', 'CO', 'NO2', 'SO2']
    devices = ['e', '3', '8', '14', '20', '30']

    fig, axes = plt.subplots(2,4, figsize=(25,10))
    axes = axes.flatten()
    print(axes)
    for i, att in enumerate(atts):
        for d in devices:
            axes[i].plot(merged['datetime'], merged[att + '_' + d], label=att + '_' + d)
        axes[i].legend()
        axes[i].set_title(att)

    # plt.gcf().autofmt_xdate()
    # plt.show()
    plt.savefig('Data/fimi_resample/envitus_fimi_overlapped.png')

def plot_correlation():
    merged = pd.read_csv("Data/fimi/envitus_fimi14_mean.csv", header=0)
    
    corr = merged.corr()
    ans = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)

    #save image 
    figure = ans.get_figure()    
    figure.savefig('img/correlations.png', dpi=800)

def merge_data():
    path = 'Data/fimi_resample/'
    names = ['fimi_1.csv', 'fimi_3.csv', 'fimi_8.csv', 'fimi_14.csv', 'fimi_20.csv', 'fimi_23.csv', 'fimi_25.csv', 'fimi_27.csv', 'fimi_30.csv']
    
    merged = pd.read_csv(path + 'envitus.csv', header=0)
    for name in names:
        fimi_a = pd.read_csv(path + name, header=0)
        merged = pd.merge(merged, fimi_a, on='datetime', how='outer')
        # merged = pd.concat([merged, fimi_a], axis=1)
    
    print(merged.head())
    merged.to_csv('Data/fimi_resample/envitus_fimi.csv', index=False)

def rename_column():
    data = pd.read_csv('Data/fimi_resample/envitus_fimi.csv', header=0)
    # i_columns = data.columns
    atts = ['PM2_5', 'PM10', 'temp', 'humidity', 'CO', 'NO2', 'SO2']
    devices = ['e', '1', '3', '8', '14', '20', '23', '25', '27', '30']
    m_columns = []
    
    for device in devices:
        for att in atts:
            m_columns.append(att + '_' + device)

    m_columns.insert(0, 'datetime')
    data.columns = m_columns

    data.to_csv('Data/fimi_resample/envitus_fimi.csv', index=False)


def test_overlap():
    data = pd.read_csv('Data/fimi_resample/envitus_fimi.csv', header=0)
    # data = data.set_index('datetime')

    e_nan_idx_null = data[data['PM2_5_e'].isnull() == False].index.tolist()
    e_nan_idx_na = data[data['PM2_5_e'].isna() == False].index.tolist()
    
    e_nan_idx_null = set(e_nan_idx_null)
    e_nan_idx_na = set(e_nan_idx_na)
    e_nan_idx = e_nan_idx_null.intersection(e_nan_idx_na)

    devices = ['1', '3', '8', '14', '20', '23', '25', '27', '30']
    
    intersects = {}
    for device in devices:
        d_nan_idx_null = data[data['PM2_5_' + device].isnull() == False].index.tolist()
        d_nan_idx_na = data[data['PM2_5_' + device].isna() == False].index.tolist()

        d_nan_idx_null = set(d_nan_idx_null)
        d_nan_idx_na = set(d_nan_idx_na)
        d_nan_idx = d_nan_idx_null.intersection(d_nan_idx_na)

        intersects[device] = len(e_nan_idx.intersection(d_nan_idx))
    
    print(intersects)

    filtered_devices = ['3', '8', '14', '20', '30']

    ovl_idx = e_nan_idx
    for device in filtered_devices:
        d_nan_idx_null = data[data['PM2_5_' + device].isnull() == False].index.tolist()
        d_nan_idx_na = data[data['PM2_5_' + device].isna() == False].index.tolist()

        d_nan_idx_null = set(d_nan_idx_null)
        d_nan_idx_na = set(d_nan_idx_na)
        d_nan_idx = d_nan_idx_null.intersection(d_nan_idx_na)

        ovl_idx = ovl_idx.intersection(d_nan_idx)
        print(len(ovl_idx))
    
    data_ovl = data.iloc[list(ovl_idx)].copy(deep=True)
    print(data_ovl.head())
    data_ovl.dropna(axis=1, how='any', inplace=True)
    data_ovl.to_csv('Data/fimi_resample/envitus_fimi_overlapped.csv', index=False)
    
if __name__ == '__main__':
    # plot_correlation()
    # names = ['fimi_1', 'fimi_3', 'fimi_8', 'fimi_14', 'fimi_20', 'fimi_23', 'fimi_25', 'fimi_27', 'fimi_30']
    # for name in names:
    #     resample_data(name + '.csv')
    
    # resample_data('envitus.csv', envitus=True)
    # merge_data()
    # rename_column()
    plot_data_new()
    # test_overlap()
