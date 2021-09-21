import pickle

import numpy as np
import pandas as pd

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=500, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def calc_kld(generated_data, ground_truth, bins, range_min, range_max):
    if range_min and range_max: 
        pd_gt, _ = np.histogram(ground_truth, bins=bins, range=(range_min, range_max), density=True)
        pd_gen, _ = np.histogram(generated_data, bins=bins, range=(range_min, range_max), density=True)
    else:
        pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True)
        pd_gen, _ = np.histogram(generated_data, bins=bins, density=True)
    kld = 0
    for x1, x2 in zip(pd_gt, pd_gen):
        if x1 != 0 and x2 == 0:
            kld += x1
        elif x1 == 0 and x2 != 0:
            kld += x2
        elif x1 != 0 and x2 != 0:
            kld += x1 * np.log(x1 / x2)

    return np.abs(kld)


def prepare_dataset(dataset, condition_size=None, use_raw=False):
    if dataset == "lorenz":
        with open("./datasets/lorenz/lorenz_dataset.pickle", "rb") as infile:
            dataset = pickle.load(infile)

        x_train = np.concatenate(list(dataset["x_train"].values()))
        y_train = np.concatenate(list(dataset["y_train"].values()))

        x_val = np.concatenate(list(dataset["x_val"].values()))
        y_val = np.concatenate(list(dataset["y_val"].values()))

        x_test = np.concatenate(list(dataset["x_test"].values()))
        y_test = np.concatenate(list(dataset["y_test"].values()))

    elif dataset == "mg":
        raw_dataset = pd.read_csv("./datasets/mg/MackyG17.csv")
        raw_dataset = np.transpose(raw_dataset.values)[0]
        print(raw_dataset.shape)
        x = [raw_dataset[i - condition_size:i] for i in range(condition_size, raw_dataset.shape[0])]
        x = np.array(x)
        y = raw_dataset[condition_size:]

        x_train = x[:int(x.shape[0] * 0.5)]
        y_train = y[:int(x.shape[0] * 0.5)]
        x_val = x[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        y_val = y[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_test = x[int(x.shape[0] * 0.6):]
        y_test = y[int(x.shape[0] * 0.6):]

    elif dataset == "itd":
        with open("./datasets/itd/a5m.pickle", "rb") as in_file:
            raw_dataset = pickle.load(in_file).astype(float)

        x = [raw_dataset[i - condition_size:i] for i in range(condition_size, raw_dataset.shape[0])]
        x = np.array(x)
        y = raw_dataset[condition_size:]

        x_train = x[:int(x.shape[0] * 0.5)]
        y_train = y[:int(x.shape[0] * 0.5)]
        x_val = x[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        y_val = y[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_test = x[int(x.shape[0] * 0.6):]
        y_test = y[int(x.shape[0] * 0.6):]

    elif dataset == 'aqm':
        raw_dataset = pd.read_csv('./Data/fimi/12_13.csv', header=0)
        raw_cols = ['PM2_5', 'temp', 'humidity']
        calib_cols = ['PM2_5_cal', 'temp_cal', 'humidity_cal']
        data_raw = raw_dataset[raw_cols].values
        data_calib = raw_dataset[calib_cols].values
        print(data_raw.shape)

        x = []
        x2 = []
        for i in range(condition_size, data_raw.shape[0]):
            if use_raw:
                x_i = data_raw[i - condition_size:i + 1, :]
            else:    
                x_i = data_calib[i - condition_size:i, :]
                x_i = np.vstack((x_i, data_raw[i: i+1, :]))
                
            x.append(x_i)
            x2.append(np.vstack((x_i, data_raw[i - condition_size: i + 1])))

        x = np.array(x)
        x2 = np.array(x2)
        y = data_calib[condition_size:]

        x_train = x[:int(x.shape[0] * 0.5)]
        x_train2 = x2[:int(x.shape[0] * 0.5)]
        y_train = y[:int(x.shape[0] * 0.5)]
        x_val = x[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_val2 = x2[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        y_val = y[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_test = x[int(x.shape[0] * 0.6):]
        x_test2 = x2[int(x.shape[0] * 0.6):]
        y_test = y[int(x.shape[0] * 0.6):]
        return x_train, x_train2, y_train, x_val, x_val2, y_val, x_test, x_test2, y_test
    
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    # x_train, y_train, x_val, y_val, x_test, y_test = prepare_dataset('mg', condition_size=6)
    # print(x_train.shape)
    x_train, x_train2, y_train, x_val, x_val2, y_val, x_test, x_test2, y_test = prepare_dataset('aqm', condition_size=6)
    print(x_train.shape)
    print(x_train2.shape)
    print(y_val.shape)