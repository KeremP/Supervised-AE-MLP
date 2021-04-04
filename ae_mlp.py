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
        self.enc_1 = GaussianNoise()
        self.enc_2 = nn.Linear(num_cols,hidden_units[0])
        self.enc_3 = nn.BatchNorm1d(hidden_units[0])
        self.silu = nn.SiLU()

        #decoder
        self.dec_1 = nn.Dropout(drop_rates[0])
        self.dec_2 = nn.Linear(hidden_units[0],num_cols)

        #ae output
        self.ae_1 = nn.Linear(num_cols,hidden_units[1])
        self.ae_2 = nn.BatchNorm1d(hidden_units[1])
        self.ae_3 = nn.Dropout(drop_rates[1])

        self.ae_out = nn.Linear(hidden_units[1],num_labels)
        self.relu = nn.ReLU()

    def forward(self, x):
        #initial batch norm
        x = self.init_bn(x)

        #encoder step
        encoder = self.enc_1(x)
        encoder = self.enc_2(encoder)
        encoder = self.enc_3(encoder)
        encoder = self.silu(encoder)

        #decoder step
        decoder = self.dec_1(encoder)
        decoder = self.dec_2(encoder)

        output = self.ae_1(decoder)
        output = self.ae_2(output)
        output = self.ae_3(output)
        output = self.ae_out(output)

        output = self.relu(output)

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
        self.init_drop = nn.Dropout(drop_rates[3])

        self.relu = nn.ReLU()


        #MLP as sequential module
        layers = []
        for i in range(2,len(hidden_units)):
            layers.append(nn.Linear(num_cols+encoder_cols,hidden_units[i]))
            layers.append(nn.BatchNorm1d(hidden_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_rates[i+2]))

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
