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
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

mse_0 = np.load('Evaluation Results/node_pd_num_dose_0.npy')
mse_1 = np.load('Evaluation Results/node_pd_num_dose_1.npy')
mse_3 = np.load('Evaluation Results/node_pd_num_dose_3.npy')
mse_4 = np.load('Evaluation Results/node_pd_num_dose_4.npy')
mse_5 = np.load('Evaluation Results/node_pd_num_dose_5.npy')
mses = np.stack([mse_0, mse_1, mse_3, mse_4, mse_5], axis=1)
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='Neural ODE', c=NODE_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=NODE_COLOR)

mse_0 = np.load('Evaluation Results/rnn_pd_num_dose_0.npy')
mse_1 = np.load('Evaluation Results/rnn_pd_num_dose_1.npy')
mse_2 = np.load('Evaluation Results/rnn_pd_num_dose_3.npy')
mse_3 = np.load('Evaluation Results/rnn_pd_num_dose_4.npy')
mse_4 = np.load('Evaluation Results/rnn_pd_num_dose_5.npy')
mses = np.stack([mse_0, mse_1, mse_3, mse_4, mse_5], axis=1)
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='RNN', c=RNN_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=RNN_COLOR)

plt.title('MSE of Concentration Prediction vs Number of Doses (PK)')
plt.xlabel('Dose Size (mg)')
plt.ylabel(r'MSE of Concentration (mg$^2$/L$^2$)')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.yticks(np.linspace(0.0, 6, 13))
plt.xticks(np.linspace(1.0, 15, 8))
plt.savefig('Figures/Num_Dose_Comparison_PD_no_mlp.pdf', format='pdf', bbox_inches='tight')

mse_0 = np.load('Evaluation Results/mlp_pd_num_dose_0.npy')
mse_1 = np.load('Evaluation Results/mlp_pd_num_dose_1.npy')
mse_2 = np.load('Evaluation Results/mlp_pd_num_dose_3.npy')
mse_3 = np.load('Evaluation Results/mlp_pd_num_dose_4.npy')
mse_4 = np.load('Evaluation Results/mlp_pd_num_dose_5.npy')
mses = np.stack([mse_0, mse_1, mse_3, mse_4, mse_5], axis=1)
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='MLP', c=MLP_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=MLP_COLOR)

plt.title('MSE of Concentration Prediction vs Number of Doses (PK)')
plt.xlabel('Dose Size (mg)')
plt.ylabel(r'MSE of Concentration (mg$^2$/L$^2$)')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.yticks(np.linspace(0.0, 9, 19))
plt.savefig('Figures/Num_Dose_Comparison_PD_with_mlp.pdf', format='pdf', bbox_inches='tight')
plt.show()
print() 