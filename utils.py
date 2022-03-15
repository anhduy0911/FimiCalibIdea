import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import wandb
import torch
import config as CFG

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

class MetricLogger:
    def __init__(self, args=None, tags=None):
        self.run = wandb.init(args.name, entity='aiotlab', config=args, tags=tags, group='Conditioned_LSTM')
        self.type = '_'.join(tags)
        self.run.name = args.name + '_' + self.type
        self.train = tags[0] == 'train'

    def log_metrics(self, step, metrics):
        self.run.log(data=metrics, step=step)

class SSA(object):

    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T

        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)

        self.TS_comps = np.zeros((self.N, self.d))

        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])

            # Diagonally average the elementary matrices, store them as columns in array.
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                X_rev = X_elem[::-1]
                self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."

            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."

        # Calculate the w-correlation matrix.
        self.calc_wcorr()

    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d

        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)

    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]

        ts_vals = self.TS_comps[:, indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)

    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """

        # Calculate the weights
        w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

        def w_inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5

        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j, i] = self.Wcorr[i, j]

    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d

        if self.Wcorr is None:
            self.calc_wcorr()

        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0, 1)

        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d - 1
        else:
            max_rnge = max

        plt.xlim(min - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min - 0.5)

    def get_lst_sigma(self):
        return self.Sigma

def get_input_data(input_file, default_n):
    dat = pd.read_csv(input_file, header=0)

    pm = dat['PM2_5'].to_list()
    temp = dat['temp'].to_list()
    hud = dat['humidity'].to_list()

    lst_pm_ssa = SSA(pm, default_n)
    lst_temp_ssa = SSA(temp, default_n)
    lst_temp_humid = SSA(hud, default_n)

    # print(lst_pm_ssa.TS_comps.shape)
    return lst_pm_ssa.TS_comps, lst_temp_ssa.TS_comps, lst_temp_humid.TS_comps

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


def prepare_dataset(condition_size=None, pred_size=0, multistep=False, ssa=False):
        raw_dataset = pd.read_csv('Data/fimi/envitus_fimi14.csv', header=0)
        raw_cols = ['PM2_5']
        calib_cols = ['PM2_5_cal']
        data_raw = raw_dataset[raw_cols].values
        data_calib = raw_dataset[calib_cols].values
        # print(data_raw.shape)
        if ssa:
            data_raw, _, _ = get_input_data('Data/fimi/envitus_fimi14.csv', 10)

        x = []
        x2 = []
        y = []
        for i in range(condition_size, data_raw.shape[0]):
            x_i = data_raw[i - condition_size:i + 1, :]
            y_i = data_calib[i - pred_size:i + 1, :]

            y.append(y_i)
            x.append(x_i)

        x = np.array(x)
        y = np.array(y)
        if not multistep: 
            y = data_calib[condition_size:]

        x_train = x[:int(x.shape[0] * 0.5)]
        y_train = y[:int(x.shape[0] * 0.5)]
        x_val = x[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        y_val = y[int(x.shape[0] * 0.5):int(x.shape[0] * 0.6)]
        x_test = x[int(x.shape[0] * 0.6):]
        y_test = y[int(x.shape[0] * 0.6):]
    
        return x_train, y_train, x_val, y_val, x_test, y_test


def prepare_multicalib_dataset(input_len=CFG.input_timestep, output_len=CFG.output_timestep, ids=CFG.devices, atts=CFG.attributes, single=False):
    dts = pd.read_csv('Data/fimi_resample/envitus_fimi_overlapped.csv', header=0)
    
    xs = []
    ys = []
    labels = []

    for idx, id in enumerate(ids):
        ls_att = ['_'.join([a, id]) for a in atts]
        print(ls_att)
        if id == 'e':
            for i in range(output_len, dts.shape[0]):
                y_i = dts[ls_att][i - output_len:i]
                ys.append(y_i)
        else:
            xis = []
            labs = []
            for i in range(input_len, dts.shape[0]):
                x_i = dts[ls_att][i - input_len:i]
                xis.append(x_i)
                
                lab = [0 for _ in range(len(ids) - 1)]
                lab[idx-1] = 1
                labs.append(lab)
            xs.append(np.array(xis))
            labels.append(labs)

    xs = np.array(xs).transpose(1, 0, 2, 3)
    ys = np.array(ys)[:, np.newaxis, :, :]
    ys = np.repeat(ys, len(ids) - 1, axis=1)
    labels = np.array(labels).transpose(1, 0, 2)

    if single:
        xs = xs.squeeze()
        ys = ys.squeeze()
        labels = labels.squeeze()

    print(xs.shape)
    print(ys.shape)
    print(labels.shape)

    x_train = xs[:int(xs.shape[0] * 0.5)]
    y_train = ys[:int(xs.shape[0] * 0.5)]
    labels_train = labels[:int(xs.shape[0] * 0.5)]
    x_val = xs[int(xs.shape[0] * 0.5):int(xs.shape[0] * 0.6)]
    y_val = ys[int(xs.shape[0] * 0.5):int(xs.shape[0] * 0.6)]
    labels_val = labels[int(xs.shape[0] * 0.5):int(xs.shape[0] * 0.6)]
    x_test = xs[int(xs.shape[0] * 0.6):]
    y_test = ys[int(xs.shape[0] * 0.6):]
    labels_test = labels[int(xs.shape[0] * 0.6):]
    return x_train, y_train, labels_train, x_val, y_val, labels_val, x_test, y_test, labels_test

def prepare_single_dataset(input_len=CFG.input_timestep, output_len=CFG.output_timestep, idex='3', atts=CFG.attributes):
    dts = pd.read_csv('Data/fimi_resample/envitus_fimi_overlapped.csv', header=0)
    
    xs = []
    ys = []
    labels = []

    ids = ['e', idex]
    for idx, id in enumerate(ids):
        ls_att = ['_'.join([a, id]) for a in atts]
        print(ls_att)
        if id == 'e':
            for i in range(output_len, dts.shape[0]):
                y_i = dts[ls_att][i - output_len:i]
                ys.append(y_i)
        else:
            for i in range(input_len, dts.shape[0]):
                x_i = dts[ls_att][i - input_len:i]
                lab = [0 for _ in range(len(ids) - 1)]
                lab[idx-1] = 1
                xs.append(x_i)
                labels.append(lab)

    xs = np.array(xs)
    ys = np.array(ys)
    labels = np.array(labels)

    print(xs.shape)
    print(ys.shape)
    print(labels.shape)

    x_train = xs[:int(xs.shape[0] * 0.5)]
    y_train = ys[:int(xs.shape[0] * 0.5)]
    labels_train = labels[:int(xs.shape[0] * 0.5)]
    x_val = xs[int(xs.shape[0] * 0.5):int(xs.shape[0] * 0.6)]
    y_val = ys[int(xs.shape[0] * 0.5):int(xs.shape[0] * 0.6)]
    labels_val = labels[int(xs.shape[0] * 0.5):int(xs.shape[0] * 0.6)]
    x_test = xs[int(xs.shape[0] * 0.6):]
    y_test = ys[int(xs.shape[0] * 0.6):]
    labels_test = labels[int(xs.shape[0] * 0.6):]
    return x_train, y_train, labels_train, x_val, y_val, labels_val, x_test, y_test, labels_test

def plot_mulidevice_dataset(ids=CFG.devices, atts=['PM2_5']):
    dts = pd.read_csv('Data/fimi_resample/envitus_fimi_overlapped.csv', header=0)
    gtruth_attr = ['_'.join([a, 'e']) for a in atts]
    gtruth = dts[gtruth_attr].values
    range_idx = range(len(gtruth)) 
    print(gtruth.shape)
    fig, ax = plt.subplots(1, len(ids), figsize=(20, 5))

    for idx, id in enumerate(ids):
        ls_att = ['_'.join([a, id]) for a in atts]
        raw_i = dts[ls_att].values
        ax[idx].plot(range_idx, raw_i, 'g', label='raw')
        ax[idx].plot(range_idx, gtruth, 'b', label='gtruth')
        ax[idx].legend(loc='best')
        ax[idx].set_title(f"device: {id}")

    fig.savefig(f"./logs/figures/multicalib_dts.png")

def plot_xtrain(xtrain, ytrain, ids=CFG.devices):
    fig, ax = plt.subplots(1, len(ids) - 1, figsize=(20, 5))
    range_idx = range(len(xtrain))
    for idx, id in enumerate(ids):
        if idx == 0:
            continue
        raw_i = xtrain[:, idx - 1, 0, 0]
        gtruth_i = ytrain[:, idx - 1, 0, 0]
        ax[idx-1].plot(range_idx, raw_i, 'g', label='raw')
        ax[idx-1].plot(range_idx, gtruth_i, 'b', label='gtruth')
        ax[idx-1].legend(loc='best')
        ax[idx-1].set_title(f"device: {id}")

    fig.savefig(f"./logs/figures/train_dts.png")

if __name__ == '__main__':
    # x_train, y_train, x_val, y_val, x_test, y_test = prepare_dataset(condition_size=6, ssa=True)
    # print(x_train.shape)
    # print(y_val.shape)

    # data_file = "Data/fimi/envitus_fimi14.csv"
    # get_input_data(data_file, 20)
    # dts = pd.read_csv('Data/fimi_resample/envitus_fimi_overlapped.csv', header=0)
    # atts = ['PM2_5_e', 'PM10_e', 'temp_e', 'humidity_e']

    # print(dts[atts][:5].values)
    x_tr, y_tr, lab_tr, _, _, _, x_ts, y_ts, _ = prepare_multicalib_dataset()
    # print(x_tr.shape)
    # print(y_tr.shape)
    # print(lab_tr.shape)
    # plot_mulidevice_dataset()
    plot_xtrain(x_ts, y_ts)
