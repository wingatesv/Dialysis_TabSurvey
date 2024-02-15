import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from models.basemodel_torch import BaseModelTorch
from utils.io_utils import get_output_path

class DeepARIMA(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.arima_order = (params["p"], params["d"], params["q"])
        self.nn_model = ARIMA_NN_Model( n_layers=self.params["n_layers"],
                                       input_dim=self.args.num_features,      
                                       hidden_dim=params["hidden_dim"],
                                       output_dim=self.args.num_classes,
                                       task=self.args.objective)

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        # Fit ARIMA model
        arima_model = ARIMA(y, order=self.arima_order)
        arima_results = arima_model.fit()

        # Extract ARIMA predictions
        arima_predictions = arima_results.predict(start=len(y), end=len(y) + len(X_val) - 1, dynamic=False)

        # Combine ARIMA predictions with neural network predictions
        combined_predictions = arima_predictions + self.nn_model.predict(X_val)

        # Other training logic for the neural network part...

        return combined_predictions  # Return combined predictions

    def predict_helper(self, X):
        # Neural network predictions
        nn_predictions = self.nn_model.predict(X)

        # ARIMA predictions (example, you may need to adjust based on your ARIMA model)
        arima_predictions = np.zeros(len(X))

        # Combine predictions
        combined_predictions = nn_predictions + arima_predictions

        return combined_predictions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "p": trial.suggest_int("p", 0, 5),  # AR order
            "d": trial.suggest_int("d", 0, 2),  # I order
            "q": trial.suggest_int("q", 0, 5),  # MA order
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        return params

class MLP(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.model = MLP_Model(n_layers=self.params["n_layers"], input_dim=self.args.num_features,
                               hidden_dim=self.params["hidden_dim"], output_dim=self.args.num_classes,
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
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        return params


class ARIMA_NN_Model(nn.Module):

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, task):
        super().__init__()

        self.task = task

        self.layers = nn.ModuleList()

        # Input Layer (= first hidden layer)
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Use ReLU as activation for all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # No activation function on the output
        x = self.output_layer(x)

        if self.task == "classification":
            x = F.softmax(x, dim=1)

        return x
