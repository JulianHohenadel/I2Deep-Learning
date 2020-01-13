import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################

        # init activation and hidden_size
        self.activation = activation
        self.hidden_size = hidden_size

        # linear layer for x and h
        self.lin_x = nn.Linear(input_size, hidden_size)
        self.lin_h = nn.Linear(hidden_size, hidden_size)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        # Initialse h as 0 if these values are not given,
        # with the correct dimsensions
        if h is None:
            h = torch.zeros((1, batch_size, hidden_size))

        # init activation
        activation = None

        # set the correct activation
        if "tanh" in self.activation:
            activation = torch.tanh
        if "relu" in self.activation:
            activation = torch.relu

        # init hidden_size
        hidden_size = self.hidden_size

        # get seq_len, batch_size, input_size from the shape of x
        seq_len, batch_size, input_size = x.shape

        # "run" the linear layers for x and h
        x = self.lin_x(x)
        h = self.lin_h(h)

        # build h_seq
        # First entry is treated seperately
        h_seq.append(activation(h.sum(0) + x[0]))

        # for each h
        # activation(linear(h at time t -1) + current x value)
        for t in range(1, seq_len):
            h_seq.append(activation(self.lin_h(h_seq[t-1]) + x[t]))

        # the wanted h is the last one from the sequence
        h = h_seq[-1]

        # concatenate h_seq
        h_seq = torch.stack(h_seq)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################

        # init everything needed for the equations
        self.hidden_size = hidden_size

        self.lin_f_x = nn.Linear(input_size, hidden_size)
        self.lin_f_h = nn.Linear(hidden_size, hidden_size)

        self.lin_i_x = nn.Linear(input_size, hidden_size)
        self.lin_i_h = nn.Linear(hidden_size, hidden_size)

        self.lin_o_x = nn.Linear(input_size, hidden_size)
        self.lin_o_h = nn.Linear(hidden_size, hidden_size)

        self.lin_c_x = nn.Linear(input_size, hidden_size)
        self.lin_c_h = nn.Linear(hidden_size, hidden_size)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        h_seq = []
        c_seq = []

        # Initialse with correct dimesions if None
        if h is None:
            h = torch.zeros((1, batch_size, hidden_size))

        if c is None:
            c = torch.zeros((1, batch_size, hidden_size))

        # get seq_len, batch_size, input_size from the shape of x
        seq_len, batch_size, input_size = x.shape

        # init hidden_size
        hidden_size = self.hidden_size

        h = h.sum(0)
        c = c.sum(0)

        # "run" the linear layers for f, i, o, c for both x and h
        f_x = self.lin_f_x(x)
        f_h = self.lin_f_h(h)

        i_x = self.lin_i_x(x)
        i_h = self.lin_i_h(h)

        o_x = self.lin_o_x(x)
        o_h = self.lin_o_h(h)

        c_x = self.lin_c_x(x)
        c_h = self.lin_c_h(h)

        f = torch.sigmoid(f_x[0] + f_h)
        i = torch.sigmoid(i_x[0] + i_h)
        o = torch.sigmoid(o_x[0] + o_h)

        # build h_seq, c_seq
        # First entry is treated seperately
        h_seq.append(o * torch.tanh(c_seq[-1]))
        c_seq.append(f * c + (i * torch.tanh(c_x[0] + c_h[0])))

        for t in range(1, seq_len):
            f_h = self.lin_f_h(h_seq[-1])
            f_t = torch.sigmoid(f_x[t] + f_h)

            i_h = self.lin_i_h(h_seq[-1])
            i_t = torch.sigmoid(i_x[t] + i_h)

            o_h = self.lin_o_h(h_seq[-1])
            o_t = torch.sigmoid(o_x[t] + o_h)

            c_h = self.lin_c_h(h_seq[-1])
            c_seq.append(f_t * c_seq[-1] + i_t * torch.tanh(c_x[t] + c_h))
            h_seq.append(o_t * torch.tanh(c_seq[-1]))

        # the wanted h, c is the last one from the sequence
        h = h_seq[-1]
        c = c_seq[-1]

        # concatenate h_seq, c_seq
        h_seq = torch.stack(h_seq)
        c_seq = torch.stack(c_seq)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        self.RNN = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        self.linear = nn.Linear(hidden_size, classes)

    def forward(self, x):

        # return h_seq, h : h is wanted
        x = self.RNN(x)[1]
        x = self.linear(x)[0]

        return x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128, num_layers=1):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, classes)

    def forward(self, x):

        # return h_seq, (h, c) : h is wanted
        x = self.LSTM(x)[1][0]
        x = self.linear(x)[0]

        return x
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
