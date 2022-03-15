import numpy as np
import torch
import torch.nn as nn
from utils import EarlyStopping, MetricLogger
from models.MulCal_v2 import MulCal
from Data.calib_loader import CalibDataset
import config as CFG
import matplotlib.pyplot as plt
class MultiCalibModel:
    def __init__(self, args, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, devices=CFG.devices, use_n=False):
        self.args = args
        self.train_loader = torch.utils.data.DataLoader(CalibDataset(x_train, y_train, lab_train), batch_size=CFG.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(CalibDataset(x_val, y_val, lab_val), batch_size=CFG.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(CalibDataset(x_test, y_test, lab_test), batch_size=CFG.batch_size, shuffle=False)
        
        self.x_test = x_test
        self.y_test = y_test

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.es = EarlyStopping(self.args.early_stop)

        print(f"Use device: {self.device}")
        print("*****  Hyper-parameters  *****")
        for k, v in vars(args).items():
            print("{}:\t{}".format(k, v))
        print("************************")

        self.model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, self.device, self.args.data_mean, self.args.data_std)
        self.model.to(self.device)
        print("\nNetwork Architecture\n")
        print(self.model)
        print("\n************************\n")
    
    def train(self):
        best_mse = np.inf

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=CFG.lr)
        criteria = nn.MSELoss()
        criteria = criteria.to(self.device)

        log_dict = {}
        logger = MetricLogger(self.args, tags=['train', 'val'])
        
        for epoch in range(CFG.epochs):
            self.model.train()
            mse_train, mae_train, mape_train = 0, 0, 0
            cnt = 0
            for x, y, lab in self.train_loader:
                cnt += 1
                x = x.to(self.device)
                y = y.to(self.device)
                lab = lab.to(self.device)
                self.model.zero_grad()

                pred = self.model(x, lab)
                # print(pred.shape)
                # print(y.shape)
                loss = criteria(pred, y)
                mae = torch.mean(torch.abs(pred - y))
                # mape = torch.mean(torch.abs((pred - y) / y)) * 100
                # print(mape)

                mse_train += loss
                mae_train += mae
                # mape_train += mape
                loss.backward()
                optimizer.step()
            
            mse_train /= cnt
            mae_train /= cnt
            # mape_train /= cnt

            log_dict['train/mse'] = mse_train
            log_dict['train/mae'] = mae_train
            # log_dict['train/mape'] = mape_train
            # validation
            self.model.eval()
            mse, mae, mape = 0, 0, 0
            cnt = 0
            for x, y, lab in self.val_loader:
                cnt += 1
                x = x.to(self.device)
                y = y.to(self.device)
                lab = lab.to(self.device)
                pred = self.model(x, lab)

                mse += criteria(pred, y)
                mae += torch.abs(pred - y).mean()
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
                torch.save(self.model.state_dict(), f"./logs/checkpoints/{self.args.name}_best.pt")
            else: 
                torch.save(self.model.state_dict(), f"./logs/checkpoints/{self.args.name}_last.pt")
                # self.model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
                if (self.es.early_stop):
                    print("Early stopping")
                    break
    
    def test(self):
        self.model.load_state_dict(torch.load(f"./logs/checkpoints/{self.args.name}_best.pt"))
        
        self.model.eval()
        mse, mae, mape = 0, 0, 0
        cnt = 0
        preds = []
        for x, y, lab in self.test_loader:
            cnt += 1
            x = x.to(self.device)
            y = y.to(self.device)
            lab = lab.to(self.device)
            pred = self.model(x, lab)

            preds.append(pred.cpu().detach().numpy())
            mse += torch.mean((pred - y) ** 2)
            mae += torch.abs(pred - y).mean()
            mape += torch.abs(pred - y).mean() / y.mean() * 100

        preds = np.concatenate(preds, axis=0)
        mse /= cnt
        mae /= cnt
        mape /= cnt

        print(f"MSE_test: {mse:.4f}, MAE_test: {mae:.4f}, MAPE_test: {mape:.4f}")
    
        ids = ['3', '8', '14', '20', '30']
        fig, ax = plt.subplots(1, len(ids), figsize=(20, 5))
        for i, idx in enumerate(ids):
            x_i = self.x_test[:, i, 0, 0]
            y_i = self.y_test[:, i, 0, 0]
            pred_i = preds[:, i, 0, 0]

            rn_test = range(x_i.shape[0])
            ax[i].plot(rn_test, x_i, 'g', label='raw')
            ax[i].plot(rn_test, y_i, 'b', label='gtruth')
            ax[i].plot(rn_test, pred_i, 'r', label='calibrated')
            ax[i].legend(loc='best')
            ax[i].set_title(f"device: {idx}")
        
        fig.savefig(f"./logs/figures/{self.args.name}_test.png")
