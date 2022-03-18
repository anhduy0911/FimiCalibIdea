from tkinter.tix import Y_REGION
from matplotlib import use
import numpy as np
import torch
import torch.nn as nn
from utils import EarlyStopping, MetricLogger
from models.MulCal_Nmodel import SingleCal
from Data.calib_loader import CalibDataset
import config as CFG
import matplotlib.pyplot as plt

class MultiCalibModel:
    def __init__(self, args, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, devices=CFG.devices, use_n=True):
        self.args = args
        self.train_loader = torch.utils.data.DataLoader(CalibDataset(x_train, y_train, lab_train), batch_size=CFG.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(CalibDataset(x_val, y_val, lab_val), batch_size=CFG.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(CalibDataset(x_test, y_test, lab_test), batch_size=CFG.batch_size, shuffle=False)
        self.use_n = use_n
        print(f'use_n: {self.use_n}')
        self.n_devices = len(devices) - 1

        self.x_test = x_test
        self.y_test = y_test


        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.es = EarlyStopping(self.args.early_stop)
        
        print(f"Use device: {self.device}")
        print("*****  Hyper-parameters  *****")
        for k, v in vars(args).items():
            print("{}:\t{}".format(k, v))
        print("************************")

        if use_n:
            self.models = [SingleCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, self.device, self.args.data_mean[i, :], self.args.data_std[i,:]).to(self.device) for i in range(self.n_devices)]
            print("\nNetwork Architecture\n")
            print(self.models[0])
            print("\n************************\n")
        else:
            self.model = SingleCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, self.device, self.args.data_mean.mean(axis=0), self.args.data_std.mean(axis=0))
            self.model.to(self.device)
            print("\nNetwork Architecture\n")
            print(self.model)
            print("\n************************\n")
    
    def train(self):
        best_mse = np.inf

        if self.use_n:
            optimizers = [torch.optim.Adam(self.models[i].parameters(), lr=CFG.lr) for i in range(self.n_devices)]
            criterias = [nn.MSELoss().to(self.device) for _ in range(self.n_devices)]
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG.lr)
            criteria = nn.MSELoss()
            criteria = criteria.to(self.device)

        log_dict = {}
        logger = MetricLogger(self.args, tags=['train', 'val'])
        
        for epoch in range(CFG.epochs):
            if self.use_n:
                for i in range(self.n_devices):
                    self.models[i].train()
            else:
                self.model.train()

            mse_train, mae_train, mape_train = 0, 0, 0
            cnt = 0
            for x, y, lab in self.train_loader:
                cnt += 1
                x = x.to(self.device)
                y = y.to(self.device)
                lab = lab.to(self.device)

                for i in range(self.n_devices):
                    input_i = x[:, i, :, :]
                    y_i = y[:, i, :, :]
                    if self.use_n:
                        self.models[i].zero_grad()
                        optimizers[i].zero_grad()
                        calib_output = self.models[i](input_i)
                        loss = criterias[i](calib_output, y_i)
                        loss.backward()
                        optimizers[i].step()
                    else:
                        self.model.zero_grad()
                        optimizer.zero_grad()
                        calib_output = self.model(input_i)
                        loss = criteria(calib_output, y_i)
                        loss.backward()
                        optimizer.step()

                mae = torch.mean(torch.abs(calib_output - y_i))
                # mape = torch.mean(torch.abs((pred - y) / y)) * 100
                # print(mape)

                mse_train += loss
                mae_train += mae
                # mape_train += mape
            
            mse_train /= cnt
            mae_train /= cnt
            # mape_train /= cnt

            log_dict['train/mse'] = mse_train
            log_dict['train/mae'] = mae_train
            # log_dict['train/mape'] = mape_train
            # validation
            if self.use_n:
                for i in range(self.n_devices):
                    self.models[i].eval()
            else:
                self.model.eval()
            mse, mae, mape = 0, 0, 0
            cnt = 0
            for x, y, lab in self.val_loader:
                if self.use_n:
                    losses = []
                cnt += 1
                x = x.to(self.device)
                y = y.to(self.device)
                lab = lab.to(self.device)

                for i in range(self.n_devices):
                    input_i = x[:, i, :, :]
                    y_i = y[:, i, :, :]
                    if self.use_n:
                        self.models[i].zero_grad()
                        calib_output = self.models[i](input_i)
                        loss = criterias[i](calib_output, y_i)
                        losses.append(loss)
                    else:
                        self.model.zero_grad()
                        calib_output = self.model(input_i)
                        loss = criteria(calib_output, y_i)

                if self.use_n:
                    print(losses)
                    loss = torch.mean(torch.stack(losses))
                
                mse += loss
                mae += torch.abs(calib_output - y_i).mean()
                # mape += torch.mean(torch.abs((pred - y) / y)) * 100
            mse /= cnt
            mae /= cnt
            # mape /= cnt

            log_dict['val/mse'] = mse
            log_dict['val/mae'] = mae
            # log_dict['val/mape'] = mape
            logger.log_metrics(epoch, log_dict)
            # print(log_dict)
            print(f"Epoch: {epoch+1:3d}/{CFG.epochs:3d}, MSE_val: {mse:.4f}, MAE_val: {mae:.4f}")
            self.es(mse)

            if mse < best_mse:
                best_mse = mse
                if self.use_n:
                    for i in range(self.n_devices):
                        torch.save(self.models[i].state_dict(), f"./logs/checkpoints/{self.args.name}_{i}_best.pt")
                else:
                    torch.save(self.model.state_dict(), f"./logs/checkpoints/{self.args.name}_best.pt")
            else: 
                if self.use_n:
                    for i in range(self.n_devices):
                        torch.save(self.models[i].state_dict(), f"./logs/checkpoints/{self.args.name}_{i}_last.pt")
                        # self.models[i].load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_{i}_best.pt"))               
                else:
                    torch.save(self.model.state_dict(), f"./logs/checkpoints/{self.args.name}_last.pt")
                    # self.model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
        
                if (self.es.early_stop):
                    print("Early stopping")
                    break
        
        self.test()
    
    def test(self):
        if self.use_n:
            for i in range(self.n_devices):
                self.models[i].load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_{i}_best.pt"))               
        else:
            self.model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
 
        if self.use_n:
            for i in range(self.n_devices):
                self.models[i].eval()
        else:
            self.model.eval()
        mse, mae, mape = 0, 0, 0
        cnt = 0
        preds = []
        for x, y, lab in self.test_loader:
            cnt += 1
            x = x.to(self.device)
            y = y.to(self.device)
            lab = lab.to(self.device)
            
            pred = []
            for i in range(self.n_devices):
                input_i = x[:, i, :, :]
                y_i = y[:, i, :, :]
                if self.use_n:
                    calib_output = self.models[i](input_i)
                    pred.append(calib_output)
                else:
                    calib_output = self.model(input_i)
                    pred.append(calib_output)

            pred = torch.stack(pred, dim=1)

            preds.append(pred.cpu().detach().numpy())
            mse += torch.mean((pred - y) ** 2)
            mae += torch.abs(pred - y).mean()
            mape += torch.abs(pred - y).mean() / y.mean() * 100

        preds = np.concatenate(preds, axis=0)
        print(f'pred shape: {preds.shape}')
        mse /= cnt
        mae /= cnt
        mape /= cnt

        print(f"MSE_test: {mse:.4f}, MAE_test: {mae:.4f}, MAPE_test: {mape:.4f}")
    
        ids = self.args.device_ids[1:]
        print(ids)
        atts = self.args.attributes
        fig, ax = plt.subplots(len(atts), len(ids), figsize=(20, 25))
        for i, idx in enumerate(ids):
            for j, att in enumerate(atts):
                x_i = self.x_test[:, i, 0, j]
                y_i = self.y_test[:, i, 0, j]
                pred_i = preds[:, i, 0, j]

                rn_test = range(x_i.shape[0])
                ax[j, i].plot(rn_test, x_i, 'g', label='raw')
                ax[j, i].plot(rn_test, y_i, 'b', label='gtruth')
                ax[j, i].plot(rn_test, pred_i, 'r', label='calibrated')
                ax[j, i].legend(loc='best')
                ax[j, i].set_title(f"device: {idx}")
                ax[j, i].set_xlabel("time")
                ax[j, i].set_ylabel(att)

        fig.savefig(f"./logs/figures/{self.args.name}_test.png")
        # fig.savefig("./logs/figures/multi_test.png")
