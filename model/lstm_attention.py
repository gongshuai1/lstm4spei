import torch
import torch.nn as nn

from torch.autograd import Variable
from einops import rearrange
from model.transformer import Transformer


class LSTM(nn.Module):

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        """
        super(LSTM, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        # Store required sizes
        self.seq_length = args.seq_length
        self.embedding_size = args.embedding_size
        self.batch_size = args.batch_size
        self.rnn_size = args.rnn_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.gru = args.gru

        self.hidden_states = Variable(torch.zeros(self.batch_size, args.rnn_size, dtype=float, device=self.args.gpu_id))
        self.cell_states = Variable(torch.zeros(self.batch_size, args.rnn_size, dtype=float, device=self.args.gpu_id))

        # The LSTM cell list
        self.cell_list = nn.ModuleList([])
        if not self.gru:
            self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size, dtype=float)
        else:
            self.cell = nn.GRUCell(self.embedding_size, self.rnn_size, dtype=float)

        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size, dtype=float)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        """
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        """
        assert x.shape[1] == self.seq_length, \
            f'input sequence length should be {self.seq_length}, but is {x.shape[1]}'

        outputs = []

        x = self.dropout(self.relu(self.input_embedding_layer(x)))  # (b, t, c)
        x = x.permute(1, 0, 2)  # (t, b, c)
        for i in range(self.seq_length):
            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(x[i], (self.hidden_states, self.cell_states))
            else:
                h_nodes = self.cell(x[i], self.hidden_states)

            outputs.append(self.relu(h_nodes))

            # Update hidden and cell states
            self.hidden_states = h_nodes.data
            if not self.gru:
                self.cell_states = c_nodes.data

        outputs = torch.stack(outputs, dim=0).permute(1, 0, 2)
        return outputs


class LSTMAttention(nn.Module):
    def __init__(self, args, infer=False):
        super().__init__()
        self.args = args
        self.lstm = LSTM(self.args, infer)
        self.transformer = Transformer(
            self.args.embedding_size, self.args.depth, self.args.heads,
            self.args.embedding_size // self.args.heads, self.args.mlp_dim, self.args.dropout)
        self.output_layer = nn.Linear(self.args.embedding_size, self.args.output_size, dtype=float)

    def forward(self, x):
        x = self.lstm(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x
