import torch
import torch.nn as nn
import config as CFG

class RGenerator(nn.Module):
    def __init__(self, noise_size, condition_size, generator_latent_size, cell_type, mean=0, std=1):
        super().__init__()
        self.noise_size = noise_size
        self.condition_size = condition_size + 1
        self.generator_latent_size = generator_latent_size
        self.mean = mean
        self.std = std

        if cell_type == "lstm":
            self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
            # self.variable_att = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5,padding=2)
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

        condition = condition.transpose(0, 1)
        variable_attention_weight, _ = self.variable_att(condition)
        variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight)

        # condition = condition.transpose(1, 2)
        # variable_attention_weight= self.variable_att(condition)
        # variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight.transpose(1, 2))

        w_condition_latent  = condition * variable_attention_weight
        condition_sum = torch.sum(w_condition_latent, dim=2)
        condition_latent, _ = self.cond_to_latent(torch.unsqueeze(condition_sum, dim=2))
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

class CCGGenerator(nn.Module):
    def __init__(self, noise_size=CFG.noise_dim, condition_size=CFG.input_dim, generator_latent_size=CFG.hidden_dim, output_timestep=CFG.output_timestep, mean=0, std=1):
        super().__init__()
        self.noise_size = noise_size
        self.condition_size = condition_size
        self.generator_latent_size = generator_latent_size
        self.output_timestep = output_timestep
        self.mean = mean
        self.std = std

        self.cond_to_latent = nn.LSTM(input_size=self.condition_size,
                                    hidden_size=self.generator_latent_size,
                                    bidirectional=False, 
                                    batch_first=True)
        
        self.latent_to_cal = nn.LSTMCell(input_size=self.generator_latent_size + self.noise_size,
                                    hidden_size=self.generator_latent_size)

        self.ctx_2_cal = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=self.generator_latent_size * 2, out_features=self.condition_size)
        ) 

    def get_warm_start(self, path=CFG.warm_start_path):
        dict = torch.load(path)
        
        self.cond_to_latent.load_state_dict(dict['cond_to_latent'])
        self.ctx_2_cal.load_state_dict(dict['ctx_2_cal'])
        
        self.cond_to_latent.requires_grad_(False)
            
    def forward(self, noise, condition):
        condition = (condition - self.mean) / self.std

        _, N, _ = condition.shape
        condition_latent, (hl, _) = self.cond_to_latent(condition)
    
        lc_input = torch.cat((hl.squeeze(0), noise), dim=1)
        condition_tilde = []
        for _ in range(self.output_timestep):
            hl_tilde, _ = self.latent_to_cal(lc_input)
            lc_input = torch.cat((hl_tilde, noise), dim=1)
            hl_coeff = torch.bmm(condition_latent, hl_tilde.unsqueeze(2)).squeeze(2)
            hl_coeff = torch.softmax(hl_coeff, dim=1)
            hl_context = torch.bmm(hl_coeff.unsqueeze(1), condition_latent).squeeze(1)

            hl_fin = torch.cat((hl_tilde, hl_context), dim=1)
            condition_tilde.append(hl_fin)

        condition_tilde = torch.stack(condition_tilde, dim=1)
        
        output = self.ctx_2_cal(condition_tilde)
        output = output * self.std + self.mean

        return output

class CCGDiscriminator(nn.Module):
    def __init__(self, condition_size=CFG.input_dim, discriminator_latent_size=CFG.hidden_dim, output_timestep=CFG.output_timestep, input_timestep=CFG.input_timestep, mean=0, std=1):
        super().__init__()
        self.condition_size = condition_size
        self.discriminator_latent_size = discriminator_latent_size
        self.half_discriminator_latent_size = discriminator_latent_size // 2
        self.mean = mean
        self.std = std

        self.input_to_latent = nn.LSTM(input_size=self.condition_size,
                                        hidden_size=discriminator_latent_size, 
                                        bidirectional=True, 
                                        batch_first=True,
                                        num_layers=1)
        
        # self.pred_to_latent = nn.LSTM(input_size=self.condition_size,
        #                                 hidden_size=discriminator_latent_size, 
        #                                 bidirectional=True, 
        #                                 batch_first=True,
        #                                 num_layers=1)
        
        self.model = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=discriminator_latent_size * 2, out_features=discriminator_latent_size),
            nn.LeakyReLU(),
            # nn.Linear(in_features=discriminator_latent_size, out_features=self.half_discriminator_latent_size),
            # nn.LeakyReLU(),
            nn.Linear(in_features=discriminator_latent_size, out_features=1),
            nn.Sigmoid()
        )
        # self.model = nn.Sequential(
        #     # nn.LeakyReLU(),
        #     nn.Linear(in_features=input_timestep * output_timestep, out_features=1),
        #     nn.Sigmoid()
        # )

    def forward(self, prediction, condition):
        condition = (condition - self.mean) / self.std
        prediction = (prediction - self.mean) / self.std
        
        _, N, _ = prediction.shape
        d_input = torch.cat((condition[:,:-N], prediction), dim=1)
        # print(d_input.shape)
        h_input, _ = self.input_to_latent(d_input)
        # h_pred, _ = self.pred_to_latent(prediction)
        
        # coeff = torch.bmm(h_input, h_pred.transpose(1,2))
        # coeff = coeff.flatten(start_dim=1).contiguous()
        # print(f'coeff: {coeff.shape}')
        # h_input, _ = self.input_to_latent(condition)
        # print(hy.shape)
        input_latent = h_input[:,-1,:]
        # pred_latent = h_pred[:,-1,:]
        # d_latent = torch.cat((input_latent, pred_latent), dim=1)
        # print(d_latent.shape)
        output = self.model(input_latent)
        
        return output

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
        
        self.variable_att = nn.LSTM(input_size=10, hidden_size=10)
        # self.variable_att = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5,padding=2)

        if cell_type == "lstm":
            self.input_to_latent = nn.LSTM(input_size=pred_dim,
                                           hidden_size=discriminator_latent_size, 
                                           bidirectional=False, 
                                           num_layers=1)
        else:
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=discriminator_latent_size)

        self.conv_dense = nn.Conv1d(in_channels=self.condition_size, out_channels=1, kernel_size=3,padding=1)
        self.model = nn.Sequential(
            nn.Linear(in_features=discriminator_latent_size, out_features=discriminator_latent_size),
            nn.ReLU(),
            nn.Linear(in_features=discriminator_latent_size, out_features=1)
        )

    def forward(self, prediction, condition):
        condition = (condition - self.mean) / self.std
        condition = condition.transpose(0, 1)
        variable_attention_weight, _ = self.variable_att(condition)
        variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight)

        # condition = condition.transpose(1, 2)
        # variable_attention_weight = self.variable_att(condition)
        # variable_attention_weight = nn.Softmax(dim=2)(variable_attention_weight.transpose(1,2))

        w_condition  = condition * variable_attention_weight
        condition_sum = torch.sum(w_condition, dim=2)

        prediction = (prediction - self.mean) / self.std
        # print(condition_sum.shape)
        # print(prediction.shape)
        d_input = torch.cat((condition_sum[:,:-1], prediction), dim=1)
        # print(d_input.size())
        # d_input = torch.cat((torch.unsqueeze(d_input, 2), condition_reshape), dim=2)
        # print(d_input.size())
        # d_input = d_input.view(-1, self.condition_size, 3)
        d_input = torch.unsqueeze(d_input, dim=2)
        d_latent, _ = self.input_to_latent(d_input)
        d_latent = self.conv_dense(d_latent.transpose(0,1))
        # d_latent = d_latent[-1]
        # print(d_latent.shape)
        output = self.model(torch.squeeze(d_latent))
        return output

if __name__ == '__main__':
    gen = CCGGenerator()
    gen.get_warm_start()
    dis = CCGDiscriminator()
    print(gen)
    print(dis)
    inp = torch.rand(128, 7, 7)
    noise = torch.rand(128, 4)

    out = gen(noise, inp)
    print(out.shape)

    out = dis(out, inp)
    print(out.shape)