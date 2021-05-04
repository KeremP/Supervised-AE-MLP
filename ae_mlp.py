import os

import torch
import torch.nn as nn
import torch.functional as F

import numpy as np
from GaussianNoise import GaussianNoise

class AE(nn.Module):
    def __init__(self, num_cols, num_labels, hidden_units, drop_rates):
        super(AE,self).__init__()

        self.num_cols = num_cols
        self.num_labels = num_labels
        self.hidden_units = hidden_units
        self.drop_rates = drop_rates

        self.init_bn = nn.BatchNorm1d(num_cols)

        #encoder
        enc_layers = []
        enc_layers.append(self.enc_1 = GaussianNoise())
        enc_layers.append(self.enc_2 = nn.Linear(num_cols,hidden_units[0]))
        enc_layers.append(self.enc_3 = nn.BatchNorm1d(hidden_units[0]))
        for i in range(len(hidden_units)-1):
            enc_layers.append(nn.Linear(hidden_units[i],hidden_units[i+1]))
            enc_layers.append(nn.BatchNorm1d(hidden_units[i+1]))
            enc_layers.append(nn.Dropout(drop_rates[i]))
        self.sequential = nn.Sequential(*enc_layers)



        #decoder
        self.dec_1 = nn.Dropout(drop_rates[-1])
        self.dec_2 = nn.Linear(hidden_units[-1],num_cols)

        #ae output
        self.ae_1 = nn.Linear(num_cols,hidden_units[1])
        self.ae_2 = nn.BatchNorm1d(hidden_units[1])
        self.ae_3 = nn.Dropout(drop_rates[1])

        self.ae_out = nn.Linear(hidden_units[1],num_labels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #initial batch norm
        x = self.init_bn(x)

        #encoder step
        encoder = self.sequential(x)

        #decoder step
        decoder = self.dec_1(encoder)
        decoder = self.dec_2(decoder)

        output = self.ae_1(decoder)
        output = self.ae_2(output)
        output = self.ae_3(output)
        output = self.ae_out(output)

        output = self.sigmoid(output)


        return output, encoder


class MLP(nn.Module):
    def __init__(self, num_cols, encoder_cols, num_labels, hidden_units, drop_rates):
        super(MLP,self).__init__()

        self.num_cols = num_cols
        self.encoder_cols = encoder_cols
        self.num_labels = num_labels
        self.hidden_units = hidden_units
        self.drop_rates = drop_rates

        self.init_bn = nn.BatchNorm1d(num_cols+encoder_cols)
        self.init_drop = nn.Dropout(drop_rates[0])

        self.relu = nn.ReLU()


        #MLP as sequential module
        layers = []
        layers.append(nn.Linear(num_cols+encoder_cols,hidden_units[0]))
        for i in range(len(hidden_units)-1):
            layers.append(nn.Linear(hidden_units[i],hidden_units[i+1]))
            layers.append(nn.BatchNorm1d(hidden_units[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_rates[i]))
        layers.append(nn.Linear(hidden_units[-1],num_labels))
        self.sequential = nn.Sequential(*layers)


    def forward(self, x, encoder_outputs):
        x = torch.cat([x,encoder_outputs],1)

        x = self.init_bn(x)
        x = self.init_drop(x)

        #MLP
        x = self.sequential(x)

        #output
        x = self.relu(x)

        return x
