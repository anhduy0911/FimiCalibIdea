import config as CFG
from torch import nn
from models.modules import *

class SingleCal(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, mean, std):
        super(SingleCal, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.half_hidden = hidden_dim // 2
        self.output_dim = output_dim
        self.device = device
        self.data_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.data_std = torch.tensor(std, dtype=torch.float32, device=device)
        print(self.data_mean)

        self.extractor_nn1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=False)
        self.extractor_nn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=False, bidirectional=True)
        
        self.lstm_ident = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=False)
        self.calib = nn.Sequential(
            nn.Linear(hidden_dim, self.half_hidden),
            nn.ReLU(),
            nn.Linear(self.half_hidden, output_dim),
        )
    
    def forward(self, input):
        # print(input.shape)
        input = (input - self.data_mean) / self.data_std
        input_ = input.permute(1, 0, 2).contiguous()
        latent_input, _ = self.extractor_nn1(input_)
        latent_input, _ = self.extractor_nn2(latent_input)
        # print(latent_input.shape)
        latent_input, _ = self.lstm_ident(latent_input)
        calib_output = self.calib(latent_input)
        calib_output = calib_output.permute(1, 0, 2).contiguous()
        calib_output = calib_output * self.data_std + self.data_mean
        
        return calib_output

if __name__ == '__main__':
    mean = torch.rand(7)
    std = torch.rand(7)
    model = SingleCal(CFG.input_dim, CFG.hidden_dim, CFG.output_dim, 'cpu', mean, std)
    print(model)

    input = torch.randn(2, 30, CFG.input_dim)
    calib_outs = model(input)

    print(calib_outs.shape)