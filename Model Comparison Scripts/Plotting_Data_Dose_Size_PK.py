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

def geo_band(mses, k=1.0):
    # mses: (n_dose, n_seed), strictly positive.
    # Build the +/- k-SD band in log space so it is asymmetric and cannot go negative.
    # Center is the geometric mean; k=1 keeps the same +/-1 SD width as before.
    log_mses = np.log(mses)
    mean_log = log_mses.mean(axis=1)
    std_log  = log_mses.std(axis=1)              # ddof=0, matches the old np.std
    center = np.exp(mean_log)                    # geometric mean
    lower  = np.exp(mean_log - k * std_log)
    upper  = np.exp(mean_log + k * std_log)
    return center, lower, upper

mse_0 = np.load('node_pk_dose_size_0_rk4_new.npy')
mse_1 = np.load('node_pk_dose_size_1_rk4_new.npy')
mse_2 = np.load('node_pk_dose_size_2_rk4_new.npy')
mse_3 = np.load('node_pk_dose_size_3_rk4_new.npy')
mse_4 = np.load('node_pk_dose_size_4_rk4_new.npy')
mses = np.stack([mse_0, mse_1, mse_2, mse_3, mse_4], axis=1)
""" center, lo, hi = geo_band(mses)
plt.plot(x, center, label='Neural ODE', c=NODE_COLOR)
plt.fill_between(x, lo, hi, alpha=0.5, color=NODE_COLOR) """
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='Neural ODE', c=NODE_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=NODE_COLOR)

mse_0 = np.load('rnn_pk_dose_size_0_new.npy')
mse_1 = np.load('rnn_pk_dose_size_1_new.npy')
mse_2 = np.load('rnn_pk_dose_size_2_new.npy')
mse_3 = np.load('rnn_pk_dose_size_3_new.npy')
mse_4 = np.load('rnn_pk_dose_size_4_new.npy')
mses = np.stack([mse_0, mse_1, mse_2, mse_3, mse_4], axis=1)
""" center, lo, hi = geo_band(mses)
plt.plot(x, center, label='RNN', c=RNN_COLOR)
plt.fill_between(x, lo, hi, alpha=0.5, color=RNN_COLOR) """
mse = np.mean(mses, axis=1)
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='RNN', c=RNN_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=RNN_COLOR)

mse_0 = np.load('mlp_pk_dose_size_0.npy')
mse_1 = np.load('mlp_pk_dose_size_1.npy')
mse_2 = np.load('mlp_pk_dose_size_2.npy')
mse_3 = np.load('mlp_pk_dose_size_3.npy')
mse_4 = np.load('mlp_pk_dose_size_4.npy')
mses = np.stack([mse_0, mse_1, mse_2, mse_3, mse_4], axis=1)
mse = np.mean(mses, axis=1)
""" center, lo, hi = geo_band(mses)
plt.plot(x, center, label='MLP', c=MLP_COLOR)
plt.fill_between(x, lo, hi, alpha=0.5, color=MLP_COLOR) """
std_mse = np.std(mses, axis=1)
plt.plot(x, mse, label='MLP', c=MLP_COLOR)
plt.fill_between(x, mse - std_mse, mse + std_mse, alpha = 0.5, color=MLP_COLOR)

plt.title('MSE of Concentration Prediction vs Dose Size (PK)')
plt.xlabel('Dose Size (mg)')
plt.ylabel(r'MSE of Concentration (mg$^2$/L$^2$)')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.yticks(np.linspace(0.0, 80, 17))
plt.savefig('Concentration_Comparison_PK.pdf', format='pdf', bbox_inches='tight')
plt.show()
print() 