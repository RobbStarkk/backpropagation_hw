import numpy as np
from typing import *


class MLP:
    def __init__(self,
                 n_inputs: int,
                 n_hidden: int,
                 hidden_sizes: Sequence[int],
                 random_state: int = 42):
        """
        Generate multilayer perceptron for regression task.
        :param n_inputs: number of inputs.
        :param n_hidden: number of hidden layers.
        :param hidden_sizes: array (may be tuple or list) with sizes of each hidden layer.
        :param random_state: seed for random. Use for initialize neurons.
        """
        self.n_inputs: int = n_inputs
        self.n_hidden: int = n_hidden
        self.hidden_sizes: Sequence[int] = hidden_sizes
        assert len(hidden_sizes) == n_hidden, 'invalid hidden sizes'
        self.seed = random_state

        np.random.seed(self.seed)
        self.layers: List[np.matrix] = []
        for i in range(n_hidden):
            if i == 0:
                d0, d1 = (n_inputs, hidden_sizes[i])
            else:
                d0, d1 = (hidden_sizes[i - 1], hidden_sizes[i])
            self.layers.append(np.matrix(np.random.rand(d0, d1)))
        d0, d1 = (hidden_sizes[len(hidden_sizes) - 1], 1)
        self.layers.append(np.matrix(np.random.rand(d0, d1)))

    def forward(self, x: Sequence, mode: str = 'eval') -> Union[float, Tuple[float, Dict[str, List[np.matrix]]]]:
        """
        Forward pass of MLP.
        :param x: Observation. 1-d array with feature values.
        :param mode: Must be “train” or “eval”. If mode is “train” then return tuple with prediction and
                     information for backprop, if mode is “eval” return prediction.
        :return: prediction float, or (prediction, {“gradient”: list of gradients,
                                                    “hidden_states”: list of hidden_states})
        """
        assert mode in ['train', 'eval'], 'mode must be train or eval'

        out = np.matrix([i for i in x])
        assert out.shape[1] == self.n_inputs, f'different sizes of x an n_inputs {out.shape[0]}, {self.n_inputs}'
        gradient: List[np.matrix] = []
        hidden_states: List[np.matrix] = [out]

        for i in range(len(self.layers)):
            out = np.dot(out, self.layers[i])
            if i != len(self.layers) - 1:
                if mode == 'train':
                    gradient.append(self.d_sigmoid(out).copy())
                out = self.sigmoid(out)
                if mode == 'train':
                    hidden_states.append(out.copy())

        if mode == 'eval':
            return float(out[0])
        else:
            return float(out[0]), {'gradient': gradient,
                                   'hidden_states': hidden_states}

    def backprop(self, y_true, y_pred, gradient: Sequence, hidden_states: Sequence, lr: float = 0.01) -> None:
        """
        Backpropagation pass for MLP.
        :param y_true: Actual label.
        :param y_pred: Prediction label.
        :param gradient: list of gradients.
        :param hidden_states: list of hidden_states.
        :param lr: learning rate.
        :return: None
        """
        # our loss for single obs. is (y_true - y_pred)^2 => d_err/d_y_pred = -2*(y_true-y_pred)
        d_err_d_pred = -2. * (y_true - y_pred)
        d_pred_d_w = hidden_states[-1]
        # change weights of output neuron
        self.layers[-1] = self.layers[-1] - (lr * d_err_d_pred * d_pred_d_w).reshape(self.layers[-1].shape)

        prev_grad = np.matrix([d_err_d_pred])
        for i in reversed(range(len(self.layers) - 1)):
            local_gradient = self._get_local_gradient(i, prev_grad)
            local_gradient = np.multiply(local_gradient, gradient[i].reshape(local_gradient.shape))
            d_err_d_w = np.dot(hidden_states[i].T, local_gradient.T)
            self.layers[i] = self.layers[i] - lr * d_err_d_w
            prev_grad = local_gradient.T

    def _get_local_gradient(self, num_layer: int, prev_grad: np.matrix):
        return np.dot(self.layers[num_layer + 1], prev_grad.T)

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def d_sigmoid(self, x):
        return np.multiply(self.sigmoid(x), (1. - self.sigmoid(x)))

    def get_layers(self, print_it: bool = True) -> Optional[Dict[str, np.matrix]]:
        """
        Print or return all layers of MLP.
        :param print_it: if True then print all layers, else return Dict with layers
        :return: None or Dict with layers.
        """
        layers_dict: Dict[str, np.matrix] = dict()
        for i in range(len(self.layers)):
            if i != len(self.layers) - 1:
                label = f"hidden layer {i + 1}"
            else:
                label = f"output layer"
            if print_it:
                print(label)
                print(self.layers[i])
            else:
                layers_dict[label] = self.layers[i]
        if not print_it:
            return layers_dict
