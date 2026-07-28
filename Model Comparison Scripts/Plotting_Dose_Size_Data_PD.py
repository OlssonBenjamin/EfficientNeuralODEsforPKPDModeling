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

x = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250])

node_mse_1 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_05_1.npy')
node_mse_2 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_05_2.npy')
node_mse_3 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_05_4.npy')
node_mse_4 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_05_4.npy')
node_mse_5 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_05_5.npy')
node_mses = np.stack([node_mse_1, node_mse_2, node_mse_3, node_mse_4, node_mse_5], axis=1)
node_mse = np.mean(node_mses, axis=1)
std_node_mse = np.std(node_mses, axis=1)
node_mse[0] = 0
std_node_mse[0] = 0
plt.plot(x, node_mse, label='Step Size of 0.5 hours', c='red')
plt.fill_between(x, node_mse - std_node_mse, node_mse + std_node_mse, alpha = 0.5, color='red')

""" node_mse_1 = np.load('flow_matching_pkpd_random_1.npy')
node_mse_2 = np.load('flow_matching_pkpd_random_2.npy')
node_mse_3 = np.load('flow_matching_pkpd_random_3.npy')
node_mse_4 = np.load('flow_matching_pkpd_random_4.npy')
node_mse_5 = np.load('flow_matching_pkpd_random_5.npy')
node_mses = np.stack([node_mse_1, node_mse_2, node_mse_3, node_mse_4, node_mse_5], axis=1)
node_mse = np.mean(node_mses, axis=1)
std_node_mse = np.std(node_mses, axis=1)
node_mse[0] = 0
std_node_mse[0] = 0
plt.plot(x, node_mse, label='PD Neural ODE trained with flow matching')
plt.fill_between(x, node_mse - std_node_mse, node_mse + std_node_mse, alpha = 0.5)

node_mse_1 = np.load('flow_matching_pkpd_cp_1.npy')
node_mse_2 = np.load('flow_matching_pkpd_cp_2.npy')
node_mse_3 = np.load('flow_matching_pkpd_cp_3.npy')
node_mse_4 = np.load('flow_matching_pkpd_cp_6.npy')
node_mse_5 = np.load('flow_matching_pkpd_cp_5.npy')
node_mses = np.stack([node_mse_1, node_mse_2, node_mse_3, node_mse_4, node_mse_5], axis=1)
node_mse = np.mean(node_mses, axis=1)
std_node_mse = np.std(node_mses, axis=1)
node_mse[0] = 0
std_node_mse[0] = 0
plt.plot(x, node_mse, label='PD Neural ODE trained with flow matching using predicted concentrations')
plt.fill_between(x, node_mse - std_node_mse, node_mse + std_node_mse, alpha = 0.5) """

node_mse_1 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_1.npy')
node_mse_2 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_2.npy')
node_mse_3 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_3.npy')
node_mse_4 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_4.npy')
node_mse_5 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_5.npy')
node_mses = np.stack([node_mse_1, node_mse_2, node_mse_3, node_mse_4, node_mse_5], axis=1)
node_mse = np.mean(node_mses, axis=1)
std_node_mse = np.std(node_mses, axis=1)
plt.plot(x, node_mse, label='Step Size of 1 hour', c='yellow')
plt.fill_between(x, node_mse - std_node_mse, node_mse + std_node_mse, alpha = 0.5, color='yellow')

node_mse_1 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_2_1.npy')
node_mse_2 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_2_2.npy')
node_mse_3 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_2_3.npy')
node_mse_4 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_2_4.npy')
node_mse_5 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_2_5.npy')
node_mses = np.stack([node_mse_1, node_mse_2, node_mse_3, node_mse_4, node_mse_5], axis=1)
node_mse = np.mean(node_mses, axis=1)
std_node_mse = np.std(node_mses, axis=1)
plt.plot(x, node_mse, label='Step Size of 2 hours', c='purple')
plt.fill_between(x, node_mse - std_node_mse, node_mse + std_node_mse, alpha = 0.5, color='purple')
 
""" node_mse_1 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_0_1.npy')
node_mse_2 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_0_2.npy')
node_mse_3 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_0_4.npy')
node_mse_4 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_0_4.npy')
node_mse_5 = np.load('No Jump Data/Node_no_jump_Dose_Size_MSE_pd_s_0_5.npy')
node_mses = np.stack([node_mse_1, node_mse_2, node_mse_3, node_mse_4, node_mse_5], axis=1)
node_mse = np.mean(node_mses, axis=1)
std_node_mse = np.std(node_mses, axis=1)
plt.plot(x, node_mse, label='No Step Size',  c='purple')
plt.fill_between(x, node_mse - std_node_mse, node_mse + std_node_mse, alpha = 0.5)  """




""" 
rnn_mse_1 = np.load('No Jump Data/RNN_No_jump_Dose_Size_MSE_pd_1.npy')
rnn_mse_2 = np.load('No Jump Data/RNN_No_jump_Dose_Size_MSE_pd_2.npy')
rnn_mse_3 = np.load('No Jump Data/RNN_No_jump_Dose_Size_MSE_pd_3.npy')
rnn_mse_4 = np.load('No Jump Data/RNN_No_jump_Dose_Size_MSE_pd_4.npy')
rnn_mse_5 = np.load('No Jump Data/RNN_No_jump_Dose_Size_MSE_pd_5.npy')
rnn_mses = np.stack([rnn_mse_1, rnn_mse_2, rnn_mse_3, rnn_mse_4, rnn_mse_5], axis=1)
rnn_mse = np.mean(rnn_mses, axis=1)
std_rnn_mse = np.std(rnn_mses, axis=1)
plt.plot(x, rnn_mse, label='RNN', c='yellow')
plt.fill_between(x, rnn_mse - std_rnn_mse, rnn_mse + std_rnn_mse, alpha = 0.5, c='yellow')  """

""" mlp_mse_1 = np.load('SimpleNet_No_Jump_Dose_Size_MSE_1.npy')
mlp_mse_2 = np.load('SimpleNet_No_Jump_Dose_Size_MSE_2.npy')
mlp_mse_3 = np.load('SimpleNet_No_Jump_Dose_Size_MSE_3.npy')
mlp_mse_4 = np.load('SimpleNet_No_Jump_Dose_Size_MSE_4.npy')
mlp_mse_5 = np.load('SimpleNet_No_Jump_Dose_Size_MSE_5.npy')
mlp_mses = np.stack([mlp_mse_1, mlp_mse_2, mlp_mse_3, mlp_mse_4, mlp_mse_5], axis=1)
mlp_mse = np.mean(mlp_mses, axis=1)
std_mlp_mse = np.std(mlp_mses, axis=1)
plt.plot(x, mlp_mse, label='MLP')
plt.fill_between(x, mlp_mse - std_mlp_mse, mlp_mse + std_mlp_mse, alpha = 0.5) """

plt.title('MSE of Concentration Prediction vs Dose Size PD')
plt.xlabel('Dose Size')
plt.ylabel('MSE of Concentration Prediction')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.yticks(np.linspace(0.0, 25, 11))
#plt.savefig('MSE vs Concentration Comparison PD')
plt.show()
print() 