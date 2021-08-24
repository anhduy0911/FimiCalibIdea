import argparse
import os

import numpy as np
import torch
import utils
from components import Generator, Discriminator
from torch import nn

# Fixing random seeds
torch.manual_seed(1368)
rs = np.random.RandomState(1368)
YELLOW_TEXT = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'


class ForGAN:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("*****  Hyper-parameters  *****")
        for k, v in vars(opt).items():
            print("{}:\t{}".format(k, v))
        print("************************")


        # Making required directories for logging, plots and models' checkpoints
        os.makedirs("./{}/".format(self.opt.dataset), exist_ok=True)

        # Defining GAN components
        self.generator = Generator(noise_size=opt.noise_size,
                                   condition_size=opt.condition_size,
                                   generator_latent_size=opt.generator_latent_size,
                                   cell_type=opt.cell_type,
                                   mean=opt.data_mean,
                                   std=opt.data_std)

        self.discriminator = Discriminator(condition_size=opt.condition_size,
                                           discriminator_latent_size=opt.discriminator_latent_size,
                                           cell_type=opt.cell_type,
                                           mean=opt.data_mean,
                                           std=opt.data_std)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        print("\nNetwork Architecture\n")
        print(self.generator)
        print(self.discriminator)
        print("\n************************\n")

    def train(self, x_train, y_train, x_val, y_val):
        x_train = torch.tensor(x_train, device=self.device, dtype=torch.float32)
        y_train = torch.tensor(y_train, device=self.device, dtype=torch.float32)
        x_val = torch.tensor(x_val, device=self.device, dtype=torch.float32)
        best_kld = np.inf
        optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.opt.lr)
        optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.opt.lr)
        adversarial_loss = nn.BCELoss()
        adversarial_loss = adversarial_loss.to(self.device)

        for step in range(self.opt.n_steps):
            d_loss = 0
            for _ in range(self.opt.d_iter):
                # train discriminator on real data
                idx = rs.choice(x_train.shape[0], self.opt.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]
                self.discriminator.zero_grad()
                d_real_decision = self.discriminator(real_data, condition)
                d_real_loss = adversarial_loss(d_real_decision,
                                               torch.full_like(d_real_decision, 1, device=self.device))
                d_real_loss.backward()
                d_loss += d_real_loss.detach().cpu().numpy()
                # train discriminator on fake data
                noise_batch = torch.tensor(rs.normal(0, 1, (condition.size(0), self.opt.noise_size)),
                                           device=self.device, dtype=torch.float32)
                x_fake = self.generator(noise_batch, condition).detach()
                d_fake_decision = self.discriminator(x_fake, condition)
                d_fake_loss = adversarial_loss(d_fake_decision,
                                               torch.full_like(d_fake_decision, 0, device=self.device))
                d_fake_loss.backward()

                optimizer_d.step()
                d_loss += d_fake_loss.detach().cpu().numpy()

            d_loss = d_loss / (2 * self.opt.d_iter)

            self.generator.zero_grad()
            noise_batch = torch.tensor(rs.normal(0, 1, (self.opt.batch_size, self.opt.noise_size)), device=self.device,
                                       dtype=torch.float32)
            x_fake = self.generator(noise_batch, condition)
            d_g_decision = self.discriminator(x_fake, condition)
            # Mackey-Glass works best with Minmax loss in our expriements while other dataset
            # produce their best result with non-saturated loss
            if opt.dataset == "mg":
                g_loss = adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 1, device=self.device))
            else:
                g_loss = -1 * adversarial_loss(d_g_decision, torch.full_like(d_g_decision, 0, device=self.device))
            g_loss.backward()
            optimizer_g.step()

            g_loss = g_loss.detach().cpu().numpy()

            # Validation
            noise_batch = torch.tensor(rs.normal(0, 1, (x_val.size(0), self.opt.noise_size)), device=self.device,
                                       dtype=torch.float32)
            preds = self.generator(noise_batch, x_val).detach().cpu().numpy().flatten()

            kld = utils.calc_kld(preds, y_val, self.opt.hist_bins, self.opt.hist_min, self.opt.hist_max)

            if kld <= best_kld and kld != np.inf:
                best_kld = kld
                print("step : {} , KLD : {}, RMSE : {}".format(step, best_kld,
                                                               np.sqrt(np.square(preds - y_val).mean())))
                torch.save({
                    'g_state_dict': self.generator.state_dict()
                }, "./{}/best.torch".format(self.opt.dataset))

            if step % 100 == 0:
                print(YELLOW_TEXT + BOLD + "step : {} , d_loss : {} , g_loss : {}".format(step, d_loss, g_loss) + ENDC)

    def test(self, x_test, y_test):
        x_test = torch.tensor(x_test, device=self.device, dtype=torch.float32)
        checkpoint = torch.load("./{}/best.torch".format(self.opt.dataset))
        self.generator.load_state_dict(checkpoint['g_state_dict'])
        y_test = y_test.flatten()
        preds = []
        rmses = []
        maes = []
        mapes = []

        for _ in range(200):
            noise_batch = torch.tensor(rs.normal(0, 1, (x_test.size(0), self.opt.noise_size)), device=self.device,
                                       dtype=torch.float32)
            pred = self.generator(noise_batch, x_test).detach().cpu().numpy().flatten()
            preds.append(pred)

            error = pred - y_test
            rmses.append(np.sqrt(np.square(error).mean()))
            maes.append(np.abs(error).mean())
            mapes.append(np.abs(error / y_test).mean() * 100)
        preds = np.vstack(preds)
        crps = np.absolute(preds[:100] - y_test).mean() - 0.5 * np.absolute(preds[:100] - preds[100:]).mean()
        preds = preds.flatten()
        kld = utils.calc_kld(preds, y_test, self.opt.hist_bins, self.opt.hist_min, self.opt.hist_max)
        print("Test resuts:\nRMSE : {}({})\nMAE : {}({})\nMAPE : {}({}) %\nCRPS : {}\nKLD : {}\n"
              .format(np.mean(rmses), np.std(rmses),
                      np.mean(maes), np.std(maes),
                      np.mean(mapes), np.std(mapes),
                      crps,
                      kld))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # mg for Mackey Glass and itd = Internet traffic dataset (A5M)
    ap.add_argument("-ds", metavar='', dest="dataset", type=str, default="lorenz",
                    help="The name of dataset: lorenz or mg or itd")
    ap.add_argument("-t", metavar='', dest="cell_type", type=str, default="gru",
                    help="The type of cells : lstm or gru")
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
    ap.add_argument("-hmin", metavar='', dest="hist_min", type=float, default=-11,
                    help="Min range of histogram for calculating KLD")
    ap.add_argument("-hmax", metavar='', dest="hist_max", type=float, default=11,
                    help="Max range of histogram for calculating KLD")

    opt = ap.parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = utils.prepare_dataset(opt.dataset, opt.condition_size)
    opt.data_mean = x_train.mean()
    opt.data_std = x_train.std()
    forgan = ForGAN(opt)
    forgan.train(x_train, y_train, x_val, y_val)
    forgan.test(x_test, y_test)
