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
        self.n_class = n_class

        self.extractor = SeriesEncoder(input_dim, hidden_dim)
        self.identity = SeriesEncoder(input_dim, hidden_dim, last_only=True)
        self.calib = IdentityAwaredCalibModule_v2(device, hidden_dim * 2, output_dim)

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
        identity_latent_inputs = []
        for i in range(M):
            input_i = input[:, i, :, :]

            input_i = (input_i - self.data_mean[i]) / self.data_std[i]
            latent_input_i = self.extractor(input_i)
            
            identity_latent_input_i = self.identity(input_i)
            identity_latent_inputs.append(identity_latent_input_i)
            # Calibration
            # latent_input_i = latent_input_i.permute(1, 0, 2).contiguous()
            calib_output = self.calib(latent_input_i, identity_latent_input_i)
            calib_output = calib_output * self.data_std[i] + self.data_mean[i]
            calib_outs.append(calib_output)

        calib_outs = torch.stack(calib_outs, dim=1)
        identity_latent_inputs = torch.stack(identity_latent_inputs, dim=1)

        return calib_outs, identity_latent_inputs


if __name__ == '__main__':
    import numpy as np
    mean = np.random.rand(5,4)
    std = np.random.rand(5,4)
    model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, 'cpu', mean, std)
    print(model)

    input = torch.randn(2, 5, 30, CFG.input_dim)
    label = torch.randn(2, 5, 5)
    calib_outs, identity_latent_inputs = model(input, label)

    print(calib_outs.shape)
    print(identity_latent_inputs.shape)