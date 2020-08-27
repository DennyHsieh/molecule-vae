import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layers, output_dim, token_order=None):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(n_hidden, output_dim)
        self.token_order = token_order

    def forward(self, x, hidden_state):
        ''' x (batch, time_step, input_size)
            hidden_state (n_layers, batch, hidden_size)
            rnn_out (batch, time_step, hidden_size) '''
        rnn_out, hidden_state = self.rnn(x, hidden_state)
        # outs = []  # save all predictions
        # for time_step in range(rnn_out.size(1)):  # calculate output for each time step
        #     outs.append(self.out(rnn_out[:, time_step, :]))
        # return torch.stack(outs, dim=1), hidden_state
        if self.token_order is None:
            return self.out(torch.sum(rnn_out, dim=1)), hidden_state
        else:
            return self.out(rnn_out[:, self.token_order, :]), hidden_state


class NetFlatten(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layers, seq_len, output_dim):
        super(NetFlatten, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(seq_len * n_hidden, output_dim)

    def forward(self, x, hidden_state):
        ''' x (batch, time_step, input_size)
            hidden_state (n_layers, batch, hidden_size)
            rnn_out (batch, time_step, hidden_size) '''
        rnn_out, hidden_state = self.rnn(x, hidden_state)
        return self.out(torch.flatten(rnn_out, start_dim=1)), hidden_state

