import numpy as np


np.random.seed(0)


class LSTMNetwork(object):
    def __init__(self):
        self.hidden_state = np.zeros((3, 3))
        self.cell_state = np.zeros((3, 3))
        self.W_hh = np.random.randn(3, 3)
        self.W_xh = np.random.randn(3, 3)
        self.W_ch = np.random.randn(3, 3)
        self.W_fh = np.random.randn(3, 3)
        self.W_ih = np.random.randn(3, 3)
        self.W_oh = np.random.randn(3, 3)
        self.W_hy = np.random.randn(3, 3)
        self.Bh = np.random.randn(3,)
        self.By = np.random.randn(3,)

    def forward_prop(self, x):
        # Input gate
        i = sigmoid(np.dot(x, self.W_xh) + np.dot(self.hidden_state, self.W_hh) + np.dot(self.cell_state, self.W_ch))
        # Forget gate
        f = sigmoid(np.dot(x, self.W_xh) + np.dot(self.hidden_state, self.W_hh) + np.dot(self.cell_state, self.W_fh))
        # Output gate
        o = sigmoid(np.dot(x, self.W_xh) + np.dot(self.hidden_state, self.W_hh) + np.dot(self.cell_state, self.W_oh))
        # New cell state
        c_new = np.tanh(np.dot(x, self.W_xh) + np.dot(self.hidden_state, self.W_hh))
        self.cell_state = f * self.cell_state + i * c_new
        self.hidden_state = o * np.tanh(self.cell_state)
        return np.dot(self.hidden_state, self.W_hy) + self.By


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


input_vector = np.ones((3, 3))
lstm = LSTMNetwork()

# Notice that the same input will lead to different outputs at each time step.
print(lstm.forward_prop(input_vector))
print(lstm.forward_prop(input_vector))
print(lstm.forward_prop(input_vector))