from torch.utils.data.dataset import TensorDataset
import utils
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd

class Generator(nn.Module):
    def __init__(self, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()

        self.condition_size = condition_size + 1
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        if cell_type == "lstm":
            self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
            # self.step_att = nn.LSTM(input_size=10, hidden_size=1)
            self.cond_to_latent = nn.LSTM(input_size=1,
                                          hidden_size=generator_latent_size,
                                          bidirectional=False)
        else:
            self.cond_to_latent = nn.GRU(input_size=self.condition_size,
                                         hidden_size=generator_latent_size, 
                                         bidirectional=True)
        self.conv_dense = nn.Conv1d(in_channels=self.condition_size, out_channels=1, kernel_size=3,padding=1)
        self.model = nn.Sequential(
            nn.Linear(in_features=generator_latent_size,
                      out_features=generator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size, out_features=1)

        )

    def forward(self, condition):
        condition = (condition - self.mean) / self.std
        # condition = condition.view(-1, self.condition_size, 1)
        # condition = condition.transpose(1, 2)
        # print(condition.size())
        # condition = self.conv_1d(condition)
        condition = condition.transpose(0, 1)
        # print(condition.size())
        # print(condition_latent.shape)
        variable_attention_weight, _ = self.variable_att(condition)
        variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight)
        # print(variable_attention_weight.shape)
        # step_attention_weight, _ = self.step_att(condition)
        # step_attention_weight = nn.Softmax(dim=1)(torch.squeeze(step_attention_weight))
        # step_attention_weight = step_attention_weight.transpose(0,1)
        # print(step_attention_weight.shape)
        w_condition_latent  = condition * variable_attention_weight
        # w_condition_latent  = condition * step_attention_weight
        # w_condition_latent = condition * torch.unsqueeze(step_attention_weight, dim=2)
        # print(w_condition_latent.shape)
        # w_condition_latent = w_condition_latent.transpose(0,1)
        condition_sum = torch.sum(w_condition_latent, dim=2)
        condition_latent, _ = self.cond_to_latent(torch.unsqueeze(condition_sum, dim=2))
        # condition_latent = condition_latent[-1]
        # print(condition_latent.shape)
        # condition_sum = condition_sum.transpose(0,1)
        # print(condition_sum.shape)
        # condition_latent, _ = self.cond_to_latent(condition_sum)
        # condition_latent = torch.squeeze(condition_latent.transpose(0, 1))
        condition_dense = self.conv_dense(condition_latent.transpose(0,1))
        # print(condition_dense.shape)
        output = self.model(torch.squeeze(condition_dense))
        output = output * self.std + self.mean

        return output