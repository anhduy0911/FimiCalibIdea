import config as CFG
import torch.nn as nn
from models.modules import *

class MulCal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_class, device, mean, std):
        super(MulCal, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)
        print(self.data_mean)
        self.n_class = n_class

        self.extractor1 = SeriesEncoder(input_dim, hidden_dim)
        self.extractor2 = SeriesEncoder(input_dim, hidden_dim)
        self.identity = IdentityLayer(hidden_dim * 2, hidden_dim, n_class)
        self.calib = IdentityAwaredCalibModule(device, hidden_dim * 2, output_dim)

    def forward(self, input):
        '''
        input with shape (N, M, L, H), in which:
            N - batch size
            M - number of device
            L - sequence length
            H - input features
        '''

        _, M, _, _ = input.shape

        calib_outs = []
        iden_outs = []
        for i in range(M):
            input_i = input[:, i, :, :]
            input_i = (input_i - self.data_mean[i]) / self.data_std[i]
            latent_input_i = self.extractor1(input_i)
            latent_input_i_2 = self.extractor2(input_i)
            identity_latent_input_i, pseudo_identity = self.identity(latent_input_i_2[-1])

            # Calibration
            # latent_input_i = latent_input_i.permute(1, 0, 2).contiguous()
            calib_output = self.calib(latent_input_i, latent_input_i_2[-1])
            calib_output = calib_output * self.data_std[i] + self.data_mean[i]
            iden_outs.append(pseudo_identity)
            calib_outs.append(calib_output)

        iden_outs = torch.stack(iden_outs, dim=1)
        calib_outs = torch.stack(calib_outs, dim=1)

        return iden_outs, calib_outs


if __name__ == '__main__':
    model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, 'cpu')
    print(model)

    input = torch.randn(2, 5, 30, CFG.input_dim)
    iden_outs, calib_outs = model(input)
    print(iden_outs.shape)
    print(calib_outs.shape)