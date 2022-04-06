import argparse
from single_calib import SingleCalibModel
import utils
# from forgan import ForGAN
# from multicalib_Nmodel import MultiCalibModel
from multicalib import MultiCalibModel
import config as CFG

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # mg for Mackey Glass and itd = Internet traffic dataset (A5M)
    ap.add_argument("-metric", metavar='', dest="metric", type=str, default='mse',
                    help="metric to save best model - mae or rmse or kld")
    ap.add_argument("-es", metavar='', dest="early_stop", type=int, default=50,
                    help="early stopping patience")
    ap.add_argument("-ssa", metavar='', dest="ssa", type=bool, default=False,
                    help="use ssa preprocessing")
    ap.add_argument("-type", metavar='', dest="train_type", type=str, default='train',
                    help="train")
    ap.add_argument("-name", metavar='', dest="name", type=str, default='multi',
                    help="name of the model - project")
    ap.add_argument("-usen", metavar='', dest="use_n", type=bool, default=False,
                    help="whether to use n model or only one")
    opt = ap.parse_args()

    # x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test = utils.prepare_multicalib_dataset()
    x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test = utils.prepare_multicalib_dataset(single=False)
   
    x_mean = x_train.mean(axis=0)
    print(x_mean.shape)
    # x_mean = x_mean.mean(axis=0)
    x_mean = x_mean.mean(axis=1)
    # print(x_mean.shape)
    x_std = x_train.std(axis=0)
    # x_std = x_std.mean(axis=0)
    x_std = x_std.mean(axis=1)
    print(x_std.shape)
    opt.data_mean = x_mean
    opt.data_std = x_std

    seed_everything(911)
    opt.device_ids = CFG.devices
    opt.attributes = CFG.attributes
    model = MultiCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test, use_n=opt.use_n)
    # model = SingleCalibModel(opt, x_train, y_train, lab_train, x_val, y_val, lab_val, x_test, y_test, lab_test)
    if opt.train_type == 'train':
        model.train()
        model.test()
    else:
        model.test()

