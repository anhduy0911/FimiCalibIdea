import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import utils
from torch import nn
from tqdm import tqdm
import pandas as pd
import config as CFG

# Fixing random seeds
torch.manual_seed(1368)
rs = np.random.RandomState(1368)
YELLOW_TEXT = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'

class CCGGenerator(nn.Module):
    def __init__(self, condition_size=CFG.input_dim, generator_latent_size=CFG.hidden_dim, output_timestep=CFG.output_timestep, mean=0, std=1):
        super().__init__()
        self.condition_size = condition_size
        self.generator_latent_size = generator_latent_size
        self.output_timestep = output_timestep
        self.mean = mean
        self.std = std

        self.cond_to_latent = nn.LSTM(input_size=self.condition_size,
                                    hidden_size=self.generator_latent_size,
                                    bidirectional=False, 
                                    batch_first=True)
        
        self.latent_to_cal = nn.LSTMCell(input_size=self.generator_latent_size,
                                    hidden_size=self.generator_latent_size)

        self.ctx_2_cal = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=self.generator_latent_size * 2, out_features=self.condition_size)
        ) 

    def forward(self, condition):
        condition = (condition - self.mean) / self.std

        _, N, _ = condition.shape
        condition_latent, (hl, _) = self.cond_to_latent(condition)
    
        lc_input = hl.squeeze(dim=0)
        condition_tilde = []
        for _ in range(self.output_timestep):
            hl_tilde, _ = self.latent_to_cal(lc_input)
            lc_input = hl_tilde
            hl_coeff = torch.bmm(condition_latent, hl_tilde.unsqueeze(2)).squeeze(2)
            hl_coeff = torch.softmax(hl_coeff, dim=1)
            hl_context = torch.bmm(hl_coeff.unsqueeze(1), condition_latent).squeeze(1)

            hl_fin = torch.cat((hl_tilde, hl_context), dim=1)
            condition_tilde.append(hl_fin)

        condition_tilde = torch.stack(condition_tilde, dim=1)
        
        output = self.ctx_2_cal(condition_tilde)
        output = output * self.std + self.mean

        return output

class CalCGenerator:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        print(f"Use device: {self.device}")
        print("*****  Hyper-parameters  *****")
        for k, v in vars(opt).items():
            print("{}:\t{}".format(k, v))
        print("************************")


        # Making required directories for logging, plots and models' checkpoints
        os.makedirs("./{}/".format(self.opt.dataset), exist_ok=True)

        # Defining GAN components
        self.generator = CCGGenerator(mean=opt.data_mean.to(self.device), std=opt.data_std.to(self.device))

        self.generator = self.generator.to(self.device)

        self.es = utils.EarlyStopping(patience=self.opt.early_stop)
        print("\nNetwork Architecture\n")
        print(self.generator)
        print("\n************************\n")

    def train(self, x_train, y_train, x_val, y_val):
        x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device, dtype=torch.float32)

        dts = TensorDataset(x_train, y_train)
        dtloader = DataLoader(dts, batch_size=self.opt.batch_size, shuffle=True)

        x_val = torch.tensor(x_val, device=self.device, dtype=torch.float32)
        best_kld = np.inf
        best_rmse = np.inf
        best_mae = np.inf

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr)
        adversarial_loss = nn.MSELoss()
        adversarial_loss = adversarial_loss.to(self.device)

        for step in tqdm(range(self.opt.n_steps)):
            g_loss = None
            for idx, (condition, y_truth) in enumerate(dtloader):
                self.generator.zero_grad()
                x_fake = self.generator(condition)
                g_loss_i = adversarial_loss(x_fake, y_truth)
                g_loss_i.backward()
                optimizer_g.step()
                if idx == 0:
                    g_loss = g_loss_i.detach().cpu().numpy()
                else:
                    g_loss += g_loss_i.detach().cpu().numpy()
            # Validation
            preds = self.generator(x_val).detach().cpu().numpy()

            rmse =  np.sqrt(np.square(preds - y_val).mean())
            mae = np.abs(preds - y_val).mean()

            if self.opt.metric == 'rmse':
                self.es(rmse)
                if self.es.early_stop:
                    break
                if rmse <= best_rmse and rmse != np.inf:
                    best_rmse = rmse
                    print("step : {}, RMSE : {}, MAE: {}".format(step,
                                                                rmse, mae))
                    torch.save({
                        'g_state_dict': self.generator.state_dict()
                    }, "./{}/best_gen.torch".format(self.opt.dataset))
            else:
                self.es(mae)
                if self.es.early_stop:
                    break
                if mae <= best_mae and mae != np.inf:
                    best_mae = mae
                    print("step : {} , RMSE : {}, MAE: {}".format(step,
                                                                rmse, mae))
                    torch.save({
                        'g_state_dict': self.generator.state_dict()
                    }, "./{}/best_gen.torch".format(self.opt.dataset))

            if step % 100 == 0:
                print(YELLOW_TEXT + BOLD + "step : {} , g_loss : {}".format(step, g_loss) + ENDC)
                torch.save({
                    'g_state_dict': self.generator.state_dict(), 
                }, "./{}/checkpoint_gen.torch".format(self.opt.dataset))

    def test(self, x_test, y_test, load_best=True):
        import os
        import matplotlib.pyplot as plt

        before_calib = x_test[:,0, 0]
        x_test = torch.tensor(x_test, device=self.device, dtype=torch.float32)
        if os.path.isfile("./{}/best_gen.torch".format(self.opt.dataset)) and load_best:
            checkpoint = torch.load("./{}/best_gen.torch".format(self.opt.dataset), map_location=self.device)
        else:
            checkpoint = torch.load("./{}/checkpoint_gen.torch".format(self.opt.dataset), map_location=self.device)
        self.generator.load_state_dict(checkpoint['g_state_dict'])

        pred = self.generator(x_test).detach().cpu().numpy()
        error = pred - y_test
        rmse = np.sqrt(np.square(error).mean())
        mae = np.abs(error).mean()
        mape = np.abs(error.mean() / y_test.mean()) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(8)
        fig.set_figwidth(22)
        fig.suptitle('PM2.5')

        ax1.plot(y_test[:,0,0], label='Real')
        ax1.plot(before_calib, label='Raw')
        ax1.set_title("Before calibration")
        ax1.legend()

        ax2.plot(y_test[:,0,0], label='Real')
        ax2.plot(pred[:,0,0], label='Prediction')
        ax2.set_title("After calibration")
        ax2.legend()
        fig.savefig('img/generator.png')

        print("Test resuts:\nRMSE : {}\nMAE : {}\nMAPE : {} %\nCRPS : {}\n"
              .format(np.mean(rmse),
                      np.mean(mae),
                      np.mean(mape),
                      mae))

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-ds", metavar='', dest="dataset", type=str, default="fimi",
                    help="The name of dataset")
    ap.add_argument("-steps", metavar='', dest="n_steps", type=int, default=10000,
                    help="Number of steps for training")
    ap.add_argument("-bs", metavar='', dest="batch_size", type=int, default=1000,
                    help="Batch size")
    ap.add_argument("-lr", metavar='', dest="lr", type=float, default=0.001,
                    help="Learning rate for RMSprop optimizer")
    ap.add_argument("-n", metavar='', dest="noise_size", type=int, default=32,
                    help="The size of Noise of Vector")
    ap.add_argument("-c", metavar='', dest="condition_size", type=int, default=24,
                    help="The size of look-back window ( Condition )")
    ap.add_argument("-rg", metavar='', dest="generator_latent_size", type=int, default=8,
                    help="The number of cells in generator")
    ap.add_argument("-rd", metavar='', dest="discriminator_latent_size", type=int, default=64,
                    help="The number of cells in discriminator")
    ap.add_argument("-d_iter", metavar='', dest="d_iter", type=int, default=2,
                    help="Number of training iteration for discriminator")
    ap.add_argument("-hbin", metavar='', dest="hist_bins", type=int, default=80,
                    help="Number of histogram bins for calculating KLD")
    ap.add_argument("-hmin", metavar='', dest="hist_min", type=float, default=None,
                    help="Min range of histogram for calculating KLD")
    ap.add_argument("-hmax", metavar='', dest="hist_max", type=float, default=None,
                    help="Max range of histogram for calculating KLD")
    ap.add_argument("-type", metavar='', dest="train_type", type=str, default='train',
                    help="train or test")
    ap.add_argument("-best", metavar='', dest="load_best", type=bool, default=True,
                    help="load best or checkpoint model")
    ap.add_argument("-metric", metavar='', dest="metric", type=str, default='kld',
                    help="metric to save best model - mae or rmse or kld")
    ap.add_argument("-es", metavar='', dest="early_stop", type=int, default=1000,
                    help="early stopping patience")
    ap.add_argument("-name", metavar='', dest="name", type=str, default='calcgan',
                    help="Name of the model")

    opt = ap.parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = utils.prepare_single_dataset()
    x_mean = x_train.mean(axis=0)
    x_mean = x_mean.mean(axis=1)
    # print(x_mean.shape)
    x_std = x_train.std(axis=0)
    x_std = x_std.mean(axis=1)
    print(x_mean.shape)
    print(x_std)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    opt.data_mean = torch.tensor(x_mean, dtype=torch.float32, device=device)
    opt.data_std = torch.tensor(x_std, dtype=torch.float32, device=device)
    
    forgan = CalCGenerator(opt)
    if opt.train_type == 'train':
        forgan.train(x_train, y_train, x_val, y_val)
        forgan.test(x_test, y_test, opt.load_best)
    else:
        forgan.test(x_test, y_test, opt.load_best)

