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

        if cell_type == "lstm":
            self.cond_to_latent = nn.LSTM(input_size=1,
                                          hidden_size=generator_latent_size)
        else:
            self.cond_to_latent = nn.GRU(input_size=1,
                                         hidden_size=generator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=generator_latent_size + self.noise_size,
                      out_features=generator_latent_size + self.noise_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size + self.noise_size, out_features=1)

        )

    def forward(self, noise, condition):
        condition = (condition - self.mean) / self.std
        condition = condition.view(-1, self.condition_size, 1)
        condition = condition.transpose(0, 1)
        condition_latent, _ = self.cond_to_latent(condition)
        condition_latent = condition_latent[-1]
        g_input = torch.cat((condition_latent, noise), dim=1)
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
            self.input_to_latent = nn.LSTM(input_size=1,
                                           hidden_size=discriminator_latent_size)
        else:
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=discriminator_latent_size)

        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, prediction, condition):
        d_input = torch.cat((condition[:, :-1], prediction.view(-1, 1)), dim=1)
       
        d_input = (d_input - self.mean) / self.std
        d_input = d_input.view(-1, self.condition_size, 1)
        d_input = d_input.transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        d_latent = d_latent[-1]
        output = self.model(d_latent)
        return output
