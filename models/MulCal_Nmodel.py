import config as CFG
import torch.nn as nn
from models.modules import *

class MulCal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_class, device, mean, std, use_n=True):
        super(MulCal, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)
        # print(self.data_mean)
        self.n_class = n_class
        self.use_n = use_n
        if self.use_n:
            self.models = nn.ModuleList([SingleCal(input_dim, hidden_dim, output_dim, device, mean[i], std[i]) for i in range(n_class)])
        else:
            mean = mean.mean(0)
            std = std.mean(0)
            self.model = SingleCal(input_dim, hidden_dim, output_dim, device, mean, std)

    def forward(self, input, label):
        '''
        input with shape (N, M, L, H), in which:
            N - batch size
            M - number of device
            L - sequence length
            H - input features
        '''

        _, M, _, _ = input.shape

        calib_outs = []
        for i in range(M):
            input_i = input[:, i, :, :]
            label_i = label[:, i, :]

            if self.use_n:
                calib_output = self.models[i](input_i)
            else:
                calib_output = self.model(input_i)

            calib_outs.append(calib_output)

        calib_outs = torch.stack(calib_outs, dim=1)

        return calib_outs

class SingleCal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, mean, std):
        super(SingleCal, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)
        print(self.data_mean)

        self.extractor = SeriesEncoder(input_dim, hidden_dim)
        self.lstm_ident = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=False)
        self.calib = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input):
        # print(input.shape)
        input = (input - self.data_mean) / self.data_std
        latent_input = self.extractor(input)
        latent_input, _ = self.lstm_ident(latent_input)
        calib_output = self.calib(latent_input)
        calib_output = calib_output.permute(1, 0, 2).contiguous()
        calib_output = calib_output * self.data_std + self.data_mean
        
        return calib_output

if __name__ == '__main__':
    mean = torch.rand(5,7)
    std = torch.rand(5,7)
    model = MulCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, CFG.n_class, 'cpu', mean, std)
    print(model)

    input = torch.randn(2, 5, 30, CFG.input_dim)
    label = torch.randn(2, 5, 5)
    calib_outs = model(input, label)

    print(calib_outs.shape)