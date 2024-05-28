import numpy as np


np.random.seed(0)


class RecurrentNetwork(object):
    """When we say W_hh, it means a weight matrix that accepts a hidden state and produce a new hidden state.
    Similarly, W_xh represents a weight matrix that accepts an input vector and produce a new hidden state. This
    notation can get messy as we get more variables later on with LSTM and I simplify the notation a little bit in
    LSTM notes.
    """
    def __init__(self):
        self.hidden_state = np.zeros((3, 3))
        self.W_hh = np.random.randn(3, 3)
        self.W_xh = np.random.randn(3, 3)
        self.W_hy = np.random.randn(3, 3)
        self.Bh = np.random.randn(3,)
        self.By = np.random.rand(3,)

    def forward_prop(self, x):
        # The order of which you do dot product is entirely up to you. The gradient updates will take care itself
        # as long as the matrix dimension matches up.
        self.hidden_state = np.tanh(np.dot(self.hidden_state, self.W_hh) + np.dot(x, self.W_xh) + self.Bh)
        return self.W_hy.dot(self.hidden_state) + self.By
    
    
input_vector = np.ones((3, 3))
rnn = RecurrentNetwork()

# Notice that same input, but leads to different ouptut at every single time step.
print(rnn.forward_prop(input_vector))
print(rnn.forward_prop(input_vector))
print(rnn.forward_prop(input_vector))