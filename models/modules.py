from turtle import forward
import torch
from torch import nn
import torch.jit as jit
from torch.nn import Parameter
from torch import Tensor
from typing import List, Tuple
from collections import namedtuple

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
class SeriesEncoder(nn.Module):
    def __init__(self, input_dim=8, output_dim=32) -> None:
        super(SeriesEncoder, self).__init__()
        # self.cnn_1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=False)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=output_dim, num_layers=1, batch_first=False, bidirectional=True)
        
    def forward(self, x):
        # x = self.cnn_1(x.permute(0, 2, 1).contiguous())
        # x = x.permute(0, 2, 1).contiguous()
        # x = self.relu(x)
        x = x.permute(1, 0, 2).contiguous()
        x, _ = self.lstm(x)
        x, _ = self.bilstm(x)
        return x


class IdentityLayer(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, nclass=5) -> None:
        super(IdentityLayer, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )
        self.classifier = nn.Linear(in_features=input_dim, out_features=nclass)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        ebd = self.module(x)
        res = self.classifier(ebd)
        res = self.softmax(res)
        return ebd, res

class IdentityLayer_v2(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, nclass=5) -> None:
        super(IdentityLayer_v2, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features=nclass, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )
        
    def forward(self, x):
        i_ebd = self.module(x)
        return i_ebd

class ConditionedLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(ConditionedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_ii = Parameter(torch.randn(4 * hidden_size, input_size))

        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.bias_ii = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, identity: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh + 
                 torch.mm(identity, self.weight_ii.t()) + self.bias_ii)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class ConditionedLSTMCellVer2(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(ConditionedLSTMCellVer2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.weight_ii = Parameter(torch.randn(hidden_size, input_size))

        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.bias_ii = Parameter(torch.randn(hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, identity: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        identgate = torch.mm(identity, self.weight_ii.t()) + self.bias_ii

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        identgate = torch.sigmoid(identgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy * identgate)

        return hy, (hy, cy)

class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class ConditionedLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, input_size, hidden_size):
        super(ConditionedLSTMLayer, self).__init__()
        self.cell = cell(input_size, hidden_size)

    @jit.script_method
    def forward(self, input: Tensor, identity: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], identity, state)
            outputs += [out]
        return torch.stack(outputs), state


class IdentityAwaredCalibModule(nn.Module):
    def __init__(self, device, input_dim=64, ouput_dim=8) -> None:
        super().__init__()
        self.hidden_dim = int(input_dim/2)
        
        self.device = device
        self.lstm = ConditionedLSTMLayer(ConditionedLSTMCell, input_size=input_dim, hidden_size=self.hidden_dim)
        self.calib = nn.Linear(in_features=self.hidden_dim, out_features=ouput_dim)
        
    def forward(self, x, i):
        _, N, _ = x.shape
        init_state = LSTMState(torch.zeros(N, self.hidden_dim).to(self.device), torch.zeros(N, self.hidden_dim).to(self.device))
        x, _ = self.lstm(x, i, init_state)
        x = self.calib(x)
        x = x.permute(1, 0, 2).contiguous()
        return x
    
class IdentityAwaredCalibModule_v2(nn.Module):
    def __init__(self, device, input_dim=64, ouput_dim=8) -> None:
        super().__init__()
        self.hidden_dim = int(input_dim/2)
        
        self.device = device
        # self.lstm = ConditionedLSTMLayer(ConditionedLSTMCell, input_size=input_dim, hidden_size=self.hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim)
        self.calib = nn.Linear(in_features=self.hidden_dim, out_features=ouput_dim)
        
    def forward(self, x, i):
        _, N, _ = x.shape
        i = i.unsqueeze(0).contiguous()
        init_state = LSTMState(i, i)
        # init_state = LSTMState(torch.zeros(N, self.hidden_dim).to(self.device), torch.zeros(N, self.hidden_dim).to(self.device))
        # x, _ = self.lstm(x, i, init_state)
        x, _ = self.lstm(x, init_state)
        x = self.calib(x)
        x = x.permute(1, 0, 2).contiguous()
        return x