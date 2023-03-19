import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .transformer import ModalEncoder

import math
from .basemodel_1D import TemporalConvNet

class SEmbeddings(nn.Module):

    def __init__(self, d_model, dim):
        super(SEmbeddings, self).__init__()
        self.lut = nn.Linear(dim, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)
        x = x * math.sqrt(self.d_model)
        return x


class TEmbeddings(nn.Module):

    def __init__(self, d_model, dim, levels=5, ksize=3, dropout=0.2):
        super(TEmbeddings, self).__init__()
        self.levels = levels
        self.ksize = ksize
        self.d_model = d_model
        self.dropout = dropout

        self.channel_sizes = [self.d_model] * self.levels
        self.lut = TemporalConvNet(dim,
                                   self.channel_sizes,
                                   kernel_size=self.ksize,
                                   dropout=self.dropout)

    def forward(self, x):
        x = self.lut(x.transpose(1, 2)).transpose(1, 2) * math.sqrt(
            self.d_model)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)
        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
class ProcessInput(nn.Module):

    def __init__(self, embed, d_model, dim, dropout_position=0.2):
        super(ProcessInput, self).__init__()

        if embed == 'spatial':
            self.Embeddings = SEmbeddings(d_model, dim)
        elif embed == 'temporal':
            self.Embeddings = TEmbeddings(d_model, dim)
        self.PositionEncoding = PositionalEncoding(d_model,
                                                   dropout_position,
                                                   max_len=5000)

    def forward(self, x):
        return self.PositionEncoding(self.Embeddings(x))

class MutilModel(nn.Module):
    def __init__(self, d_in, d_model, len_feature, ntarget=7, embed='temporal') -> None:
        super(MutilModel, self).__init__()
        self.modal_num = len(d_in)
        self.d_model = d_model
        self.num_features = d_in

        self.input = nn.ModuleList()
        for i in range(self.modal_num):
            self.input.append(ProcessInput(embed, self.d_model, self.num_features[i]))

        self.modality_encoders = nn.ModuleList()
        for _ in range(self.modal_num):
            self.modality_encoders.append(
                ModalEncoder(dim=self.d_model, dropout=0.5))

        self.modal_interaction = ModalEncoder(
            dim=self.d_model * self.modal_num,
            heads=8
        )

        self.regress = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * self.d_model * self.modal_num * len_feature,
                      self.d_model * self.modal_num // 2),
            nn.ReLU(inplace=True)

        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * self.d_model * self.modal_num * len_feature,
                      self.d_model * self.modal_num // 2),
            nn.ReLU(inplace=True)
        )

        self.intensity_layer = nn.Linear(
            self.d_model * self.modal_num, ntarget)
        self.class_layer = nn.Linear(
            self.d_model * self.modal_num, ntarget)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        _x_list = []
        _x_embed = []
        for i in range(self.modal_num):
            x_embed = self.input[i](x[i])
            _x_embed.append(x_embed)
            _x_list.append(self.modality_encoders[i](x_embed))
        x = torch.concat(_x_list, dim=-1)
        x = self.regress(x)

        _x_embed = torch.concat(_x_embed, dim=-1)
        av = self.proj(self.modal_interaction(_x_embed))

        x = torch.concat([x, av], dim=1)
        preds1 = self.final_activation(self.intensity_layer(x))
        preds2 = self.class_layer(x)
        return preds1, preds2

