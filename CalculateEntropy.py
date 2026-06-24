import torchdiffeq as tdq
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import copy

class SWAGTracker:
    def __init__(self):
        self.snapshots = []
        self.mean_param = None
        self.cov_param = None

    def collect_model(self, model):
        self.snapshots.append(copy.deepcopy(model.state_dict()))

    def finalize(self):
        if len(self.snapshots) == 0:
            raise ValueError("No snapshots collected for SWAG.")
        param_vectors = []
        for state_dict in self.snapshots:
            vec = []
            for param_tensor in state_dict.values():
                vec.append(param_tensor.view(-1))
            vec = torch.cat(vec)
            param_vectors.append(vec.unsqueeze(0))
        param_matrix = torch.cat(param_vectors, dim=0)
        self.mean_param = param_matrix.mean(dim=0)
        centered = param_matrix - self.mean_param.unsqueeze(0)
        self.cov_param = (centered.t() @ centered) / (len(self.snapshots) - 1)

        # Adding a small constant to the diagonal for numerical stability

        return self.mean_param, self.cov_param

    def sample(self):
        if self.mean_param is None or self.cov_param is None:
            raise ValueError("Call finalize() before sampling.")
        num_params = self.mean_param.shape[0]

        eps = torch.randn(num_params)
        L = torch.linalg.cholesky(self.cov_param)
        sample_vec = self.mean_param + L @ eps
        return sample_vec

    def calculate_entropy(self):
        if self.cov_param is None:
            raise ValueError("Covariance matrix not available. Finalize first.")
        
        self.cov_param += 1e-6 * torch.eye(self.cov_param.size(0))
        num_params = self.cov_param.size(0)

        L = torch.linalg.cholesky(self.cov_param)
        log_det_cov = 2 * torch.sum(torch.log(torch.diag(L)))

        entropy = 0.5 * (log_det_cov + num_params * (1 + np.log(2 * np.pi)))
        
        return entropy

swag_tracker = SWAGTracker()
swag_mean = torch.load('swag_mean_exponential_swag_0005_tan_1.pth')
swag_cov = torch.load('swag_cov_exponential_swag_0005_tan_1.pth')
swag_tracker.mean_param = swag_mean
swag_tracker.cov_param = swag_cov
entropy = swag_tracker.calculate_entropy()
print(entropy)