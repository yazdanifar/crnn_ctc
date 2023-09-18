import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, rnn_hidden_size, rnn_num_layers, channels):
        super(CRNN, self).__init__()
        self.cnn_out_height = 7
        self.cnn_out_width = 22
        self.last_conv_channel = channels[-1]
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(self.last_conv_channel),
            nn.ReLU()
        )
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_input_size = self.cnn_out_height * self.last_conv_channel
        self.rnn = nn.GRU(self.rnn_input_size, rnn_hidden_size, rnn_num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Conv1d(in_channels=self.rnn_hidden_size * 2, out_channels=11, kernel_size=1)

    def forward(self, x):
        x = self.conv_params(x)  # b,c,h,w
        x = x.permute(0, 3, 2, 1)  # b,w,h,c
        x = x.reshape(-1, self.cnn_out_width, self.rnn_input_size)
        x, _ = self.rnn(x)  # b, w, rnn_input_size
        x = x.permute(0, 2, 1)  # b, rnn_hidden_size, w
        x = self.classifier(x)
        x = x.permute(2, 0, 1)
        x = F.log_softmax(x, dim=-1)

        return x
