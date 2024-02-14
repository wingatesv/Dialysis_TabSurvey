import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from models.basemodel_torch import BaseModelTorch
from utils.io_utils import get_output_path

class RNN(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.model = RNN_Model(input_dim=self.args.num_features,
                               hidden_dim=params["hidden_dim"],
                               output_dim=self.args.num_classes,
                               num_layers=params["num_layers"],
                               task=self.args.objective)

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        return params


class RNN_Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, task):
        super().__init__()

        self.task = task

        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        # Take the output from the last time step
        rnn_out = rnn_out[:, -1, :]
        output = self.fc(rnn_out)

        if self.task == "classification":
            output = F.softmax(output, dim=1)

        return output
