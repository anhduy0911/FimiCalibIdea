import torch
import torch.nn as nn

class RGenerator(nn.Module):
    def __init__(self, noise_size, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()
        self.noise_size = noise_size
        self.condition_size = condition_size + 1
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        if cell_type == "lstm":
            # self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
            self.variable_att = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5,padding=2)
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
            nn.Linear(in_features=generator_latent_size + self.noise_size,
                      out_features=generator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size, out_features=1)

        )

    def forward(self, noise, condition):
        condition = (condition - self.mean) / self.std

        # condition = condition.transpose(0, 1)
        # variable_attention_weight, _ = self.variable_att(condition)
        # variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight)

        condition = condition.transpose(1, 2)
        variable_attention_weight= self.variable_att(condition)
        variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight.transpose(1, 2))

        w_condition_latent  = condition.transpose(1,2) * variable_attention_weight
        condition_sum = torch.sum(w_condition_latent, dim=2)
        condition_latent, _ = self.cond_to_latent(torch.unsqueeze(condition_sum, dim=2).transpose(0, 1))
        # condition_latent = condition_latent[-1]
        # print(condition_latent.shape)
        # condition_sum = condition_sum.transpose(0,1)
        # print(condition_sum.shape)
        # condition_latent, _ = self.cond_to_latent(condition_sum)
        # condition_latent = torch.squeeze(condition_latent.transpose(0, 1))
        condition_dense = self.conv_dense(condition_latent.transpose(0,1))
        g_input = torch.cat((torch.squeeze(condition_dense), noise), dim=1)
        output = self.model(g_input)
        output = output * self.std + self.mean

        return output

class AttGenerator(nn.Module):
    def __init__(self, noise_size, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()

        self.condition_size = condition_size + 1
        self.noise_size = noise_size
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        self.variable_att = nn.Linear(in_features=self.condition_size, out_features=1)

        if cell_type == "lstm":
            # self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
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
            nn.Linear(in_features=generator_latent_size + self.noise_size,
                      out_features=generator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size, out_features=1)
        )

    def forward(self,noise, condition):
        condition = (condition - self.mean) / self.std
        # condition = condition.transpose(0, 1)

        weights = []
        for i in range(10):
            weights.append(self.variable_att(condition[:, :, i]))
        # variable_attention_weight, _ = self.variable_att(condition)
        weights = torch.cat(weights, 1)
        # print(weights.shape)

        variable_attention_weight = nn.Softmax(dim=1)(weights)
        # print(variable_attention_weight.shape)
        # step_attention_weight, _ = self.step_att(condition)
        # step_attention_weight = nn.Softmax(dim=1)(torch.squeeze(step_attention_weight))
        # step_attention_weight = step_attention_weight.transpose(0,1)
        # print(step_attention_weight.shape)
        w_condition_latent  = condition * torch.unsqueeze(variable_attention_weight, 1)
        # w_condition_latent  = condition * step_attention_weight
        # w_condition_latent = condition * torch.unsqueeze(step_attention_weight, dim=2)
        # print(w_condition_latent.shape)
        # w_condition_latent = w_condition_latent.transpose(0,1)
        condition_sum = torch.sum(w_condition_latent, dim=2)
        condition_latent, _ = self.cond_to_latent(torch.unsqueeze(condition_sum, dim=2).transpose(0,1))
        # condition_latent = condition_latent[-1]
        # print(condition_latent.shape)
        # condition_sum = condition_sum.transpose(0,1)
        # print(condition_sum.shape)
        # condition_latent, _ = self.cond_to_latent(condition_sum)
        # condition_latent = torch.squeeze(condition_latent.transpose(0, 1))
        condition_dense = self.conv_dense(condition_latent.transpose(0,1))
        g_input = torch.cat((torch.squeeze(condition_dense), noise), dim=1)
        # print(condition_dense.shape)
        output = self.model(g_input)
        output = output * self.std + self.mean

        return output

class Generator(nn.Module):
    def __init__(self, noise_size, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()

        self.noise_size = noise_size
        self.condition_size = condition_size + 1
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        self.conv_1d = nn.Conv1d(in_channels=3, out_channels=generator_latent_size, kernel_size=3, padding=1)
        if cell_type == "lstm":
            self.cond_to_latent = nn.LSTM(input_size=self.condition_size,
                                          hidden_size=generator_latent_size,
                                          bidirectional=True)
        else:
            self.cond_to_latent = nn.GRU(input_size=self.condition_size,
                                         hidden_size=generator_latent_size, 
                                         bidirectional=True)
        self.conv_dense = nn.Conv1d(in_channels=generator_latent_size, out_channels=1, kernel_size=3,padding=1)
        self.model = nn.Sequential(
            nn.Linear(in_features=generator_latent_size*2 + self.noise_size,
                      out_features=generator_latent_size + self.noise_size),
            nn.ReLU(),
            nn.Linear(in_features=generator_latent_size + self.noise_size, out_features=1)

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
        # condition_latent = condition_latent[-1]
        condition_dense = self.conv_dense(condition_latent.transpose(0,1))
        g_input = torch.cat((torch.squeeze(condition_dense), noise), dim=1)
        # print(g_input.shape)
        output = self.model(g_input)
        output = output * self.std + self.mean

        return output

    def get_noise_size(self):
        return self.noise_size

class AttDiscriminator(nn.Module):
    def __init__(self, condition_size, discriminator_latent_size, cell_type, pred_dim = 1, mean=0, std=1):
        super().__init__()
        self.discriminator_latent_size = discriminator_latent_size
        self.condition_size = condition_size + 1
        self.mean = mean
        self.std = std
        
        self.cond_to_z = nn.LSTM(input_size=10,
                                    hidden_size=pred_dim, 
                                    bidirectional=False, 
                                    num_layers=1)
        
        # self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
        self.variable_att = nn.Linear(in_features=self.condition_size, out_features=1)
        
        if cell_type == "lstm":
            self.input_to_latent = nn.LSTM(input_size=pred_dim,
                                           hidden_size=discriminator_latent_size, 
                                           bidirectional=True, 
                                           num_layers=1)
        else:
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=discriminator_latent_size)

        # self.conv_dense = nn.Conv1d(in_channels=self.condition_size, out_channels=1, kernel_size=3,padding=1)
        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size*2, out_features=discriminator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=discriminator_latent_size, out_features=1)
        )

    def forward(self, prediction, condition):
        condition = (condition - self.mean) / self.std
        # condition = condition.transpose(0, 1)
        # variable_attention_weight, _ = self.variable_att(condition)
        # variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight)
        weights = []
        for i in range(10):
            weights.append(self.variable_att(condition[:, :, i]))
        # variable_attention_weight, _ = self.variable_att(condition)
        weights = torch.cat(weights, 1)

        variable_attention_weight = nn.Softmax(dim=1)(weights)
        w_condition  = condition * torch.unsqueeze(variable_attention_weight, 1)
        condition_sum = torch.sum(w_condition, dim=2)

        prediction = (prediction - self.mean) / self.std
        # print(condition_sum.shape)
        # print(prediction.shape)
        d_input = torch.cat((condition_sum[:,:-1], prediction), dim=1)
        # print(d_input.size())
        # d_input = torch.cat((torch.unsqueeze(d_input, 2), condition_reshape), dim=2)
        # print(d_input.size())
        # d_input = d_input.view(-1, self.condition_size, 3)
        d_input = torch.unsqueeze(d_input, dim=2).transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        # d_latent = self.conv_dense(d_latent.transpose(0,1))
        d_latent = d_latent[-1]
        # print(d_latent.shape)
        output = self.model(torch.squeeze(d_latent))
        return output

class Discriminator(nn.Module):
    def __init__(self, condition_size, discriminator_latent_size, cell_type, pred_dim = 1, mean=0, std=1):
        super().__init__()
        self.discriminator_latent_size = discriminator_latent_size
        self.condition_size = condition_size + 1
        self.mean = mean
        self.std = std
        
        self.cond_to_z = nn.LSTM(input_size=10,
                                    hidden_size=pred_dim, 
                                    bidirectional=False, 
                                    num_layers=1)
        
        # self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
        self.variable_att = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5,padding=2)

        if cell_type == "lstm":
            self.input_to_latent = nn.LSTM(input_size=pred_dim,
                                           hidden_size=discriminator_latent_size, 
                                           bidirectional=True, 
                                           num_layers=1)
        else:
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=discriminator_latent_size)

        # self.conv_dense = nn.Conv1d(in_channels=self.condition_size, out_channels=1, kernel_size=3,padding=1)
        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size*2, out_features=discriminator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=discriminator_latent_size, out_features=1)
        )

    def forward(self, prediction, condition):
        condition = (condition - self.mean) / self.std
        # condition = condition.transpose(0, 1)
        # variable_attention_weight, _ = self.variable_att(condition)
        # variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight)

        condition = condition.transpose(1, 2)
        variable_attention_weight = self.variable_att(condition)
        variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight.transpose(1,2))

        w_condition  = condition.transpose(1,2) * variable_attention_weight
        condition_sum = torch.sum(w_condition, dim=2)

        prediction = (prediction - self.mean) / self.std
        # print(condition_sum.shape)
        # print(prediction.shape)
        d_input = torch.cat((condition_sum[:,:-1], prediction), dim=1)
        # print(d_input.size())
        # d_input = torch.cat((torch.unsqueeze(d_input, 2), condition_reshape), dim=2)
        # print(d_input.size())
        # d_input = d_input.view(-1, self.condition_size, 3)
        d_input = torch.unsqueeze(d_input, dim=2).transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        # d_latent = self.conv_dense(d_latent.transpose(0,1))
        d_latent = d_latent[-1]
        # print(d_latent.shape)
        output = self.model(torch.squeeze(d_latent))
        return output
