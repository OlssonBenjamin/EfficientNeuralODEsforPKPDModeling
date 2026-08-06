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

NODE_COLOR = '#0072B2'  
RNN_COLOR  = '#E69F00'  
MLP_COLOR = '#009E73'
x = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])

mse_0 = np.load('node_pd_dose_size_0_1.npy')
mse_1 = np.load('node_pd_dose_size_1_1.npy')
mse_2 = np.load('node_pd_dose_size_2_1.npy')
mse_3 = np.load('node_pd_dose_size_3_1.npy')
mse_4 = np.load('node_pd_dose_size_4_1.npy')
mses = np.stack([mse_0, mse_1, mse_2, mse_3, mse_4], axis=1)
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='Neural ODE', c=NODE_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=NODE_COLOR)

mse_0 = np.load('rnn_pd_dose_size_0.npy')
mse_1 = np.load('rnn_pd_dose_size_1.npy')
mse_2 = np.load('rnn_pd_dose_size_2.npy')
mse_3 = np.load('rnn_pd_dose_size_3.npy')
mse_4 = np.load('rnn_pd_dose_size_4.npy')
mses = np.stack([mse_0, mse_1, mse_2, mse_3, mse_4], axis=1)
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='RNN', c=RNN_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=RNN_COLOR)

plt.title('MSE of Concentration Prediction vs Dose Size (PD)')
plt.xlabel('Dose Size (mg)')
plt.ylabel(r'MSE of Response Prediction$^2$')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.yticks(np.linspace(0.0, 60, 13))
plt.savefig('Concentration_Comparison_PD_no_MLP.pdf', format='pdf', bbox_inches='tight')

mse_0 = np.load('mlp_pd_dose_size_0.npy')
mse_1 = np.load('mlp_pd_dose_size_1.npy')
mse_2 = np.load('mlp_pd_dose_size_2.npy')
mse_3 = np.load('mlp_pd_dose_size_3.npy')
mse_4 = np.load('mlp_pd_dose_size_4.npy')
mses = np.stack([mse_0, mse_1, mse_2, mse_3, mse_4], axis=1)
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='MLP', c=MLP_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=MLP_COLOR)
plt.yticks(np.linspace(0.0, 600, 13))
plt.savefig('Concentration_Comparison_PD_with_MLP.pdf', format='pdf', bbox_inches='tight')
print() 