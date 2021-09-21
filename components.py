import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()

        self.noise_size = noise_size
        self.condition_size = condition_size + 1
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        self.conv_1d = nn.Conv1d(in_channels=3, out_channels=generator_latent_size, kernel_size=3)
        if cell_type == "lstm":
            self.cond_to_latent = nn.LSTM(input_size=6,
                                          hidden_size=generator_latent_size,
                                          bidirectional=True)
        else:
            self.cond_to_latent = nn.GRU(input_size=6,
                                         hidden_size=generator_latent_size, 
                                         bidirectional=True)

        self.model = nn.Sequential(
            nn.Linear(in_features=generator_latent_size*2 + self.noise_size,
                      out_features=generator_latent_size + self.noise_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size + self.noise_size, out_features=3)

        )

    def forward(self, noise, condition):
        condition = (condition - self.mean) / self.std
        # condition = condition.view(-1, self.condition_size, 1)
        condition = condition.transpose(1, 2)
        # print(condition.size())
        condition = self.conv_1d(condition)
        # print(condition.size())
        condition = condition.transpose(0, 1)
        condition_latent, _ = self.cond_to_latent(condition)
        condition_latent = condition_latent[-1]
        g_input = torch.cat((condition_latent, noise), dim=1)
        # print(g_input.shape)
        output = self.model(g_input)
        output = output * self.std + self.mean

        return output

    def get_noise_size(self):
        return self.noise_size


class Discriminator(nn.Module):
    def __init__(self, condition_size, discriminator_latent_size, cell_type, mean=0, std=1):
        super().__init__()
        self.discriminator_latent_size = discriminator_latent_size
        self.condition_size = condition_size + 1
        self.mean = mean
        self.std = std

        if cell_type == "lstm":
            self.input_to_latent = nn.LSTM(input_size=3,
                                           hidden_size=discriminator_latent_size, 
                                           bidirectional=True, 
                                           num_layers=3)
        else:
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=discriminator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size*2, out_features=discriminator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=discriminator_latent_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, prediction, condition):
        d_input = torch.cat((condition[:, :-1], prediction.view(-1, 1, 3)), dim=1)
        d_input = (d_input - self.mean) / self.std
        # print(d_input.size())
        d_input = d_input.view(-1, self.condition_size, 3)
        d_input = d_input.transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        d_latent = d_latent[-1]
        output = self.model(d_latent)
        return output
