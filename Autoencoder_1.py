import numpy as np
import torch
import torch.nn as nn
from Solver import Solver


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size,
                             num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size,
                             num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        out, (hn, cn) = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (hn, cn))
        return out


class AutoEncoder(nn.Module):

    def __init__(self, feature_size, output_size, num_lstm_layers=2):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(feature_size, output_size, num_lstm_layers)

    def forward(self, x):
        encoding = self.encoder(x)
        encoding = encoding[:, 0, :, :]
        y_pred = self.decoder(encoding)
        return y_pred

def create_model(device, train_loader):
    epoch = 100
    lr = 1e-4
    feature_size = 8 #x_train.shape[3]
    output_size = 8 #x_train.shape[3]
    num_lstm_layers = 2
    batch_size = 16
    model = AutoEncoder(feature_size, output_size, num_lstm_layers)
    model.apply(initialize_parameters)
    optim = torch.optim.Adam(model.parameters())
    criterion = nn.L1Loss()
    model = Solver(device, model, train_loader, optim, criterion, epoch=epoch, lr=lr, print_every=1)
    torch.save(model, 'ae1-model.pth')
    print('Model created and saved successfully')
