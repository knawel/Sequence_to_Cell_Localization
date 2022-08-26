import torch.nn as nn
import torch as pt
from torch.autograd import Variable

_ = pt.manual_seed(142)
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dev):
        super(RNN, self).__init__()
        self.hidden2 = 64
        self.hidden3 = 64
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_size, self.hidden2)
        self.fc2 = nn.Linear(self.hidden2, self.hidden3)
        self.fc3 = nn.Linear(self.hidden3, num_classes)
        self.relu = nn.ReLU()
        # self.hidden2tag = nn.Linear(self.hidden_size, num_classes)
        self.device = dev

    def forward(self, x):
        x = x.float()
        h0 = Variable(pt.zeros(self.num_layers, x.size(0), self.hidden_size).float()).to(self.device)
        c0 = Variable(pt.zeros(self.num_layers, x.size(0), self.hidden_size).float()).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        tag_scores = self.fc3(out)
        # tag_space = self.hidden2tag(out[:, -1, :])
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


