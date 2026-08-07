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

class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ODEFunc(nn.Module):
   def __init__(self, bottleneck):
       super(ODEFunc, self).__init__()
       self.net = nn.Sequential(
           nn.Linear(bottleneck * 2 + 1, 32),
           nn.SiLU(),
           nn.Linear(32, 32),
           nn.SiLU(),
           nn.Linear(32, 1)
       )
       self.dose_encoding = nn.Sequential(
           nn.Linear(2, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )

       self.bottleneck = bottleneck

   def forward(self, t, y):
        y0 = y[:, 0].unsqueeze(1)
        steering_variables = y[:, 1:self.bottleneck + 1]
        dosing_times = y[:, self.bottleneck + 1:-1]
        dose_size = y[:, -1].unsqueeze(-1)
        t_expanded = t.unsqueeze(0).expand((y0.size(0), dosing_times.size(1)))
        mask = (dosing_times <= t_expanded)
        masked_dosing_times = dosing_times * mask 
        last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]
        dose_input = torch.cat([last_dose, dose_size], dim=1)
        dose_encoding = self.dose_encoding(dose_input)
        ode_input = torch.cat([y0, steering_variables, dose_encoding], dim=1)
        dzdt = self.net(ode_input)  
        room_up   = (50.0 - y0) / 50.0           
        room_down =  y0 / 50.0
        sign_mask = dzdt > 0 
        time_mask = last_dose > 24.0 
        dzdt = torch.where(sign_mask & time_mask, dzdt * room_up, dzdt)  
        dzdt = torch.where(~sign_mask, dzdt * room_down, dzdt)    
        dzdt = torch.cat([dzdt, torch.zeros_like(steering_variables), torch.zeros_like(dosing_times), torch.zeros_like(dose_size)], dim=1)
        return dzdt
                

class NeuralODE(nn.Module):
   def __init__(self, ode_func):
       super(NeuralODE, self).__init__()
       self.ode_func = ode_func

   def forward(self, input, t):
       return odeint(self.ode_func, input, t, method='euler', options=dict(step_size=2.0))

class LatentODE(nn.Module):
    def __init__(self, num_variables, node, bottleneck):
       super(LatentODE, self).__init__()
       self.patient_encoding = nn.Sequential(
           nn.Linear(num_variables, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )

       self.node = node

    def forward(self, input, dosing_times, predict_times):
        x = self.patient_encoding(input[:, :-1])
        batch_size = x.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32) + 50
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        return latent_time_series[:, :, 0]

def load_validation_data(data_file, covariate_file, dose_size, dose_regime_index):
    pk_pd_dataframe = pd.read_csv(data_file)
    cp_response_data = pk_pd_dataframe[['ID', 'time', 'CP', 'RESPONSE']]
    grouped_data = cp_response_data.groupby('ID')
    individual_patient_data = {}
    for patient_id, group in grouped_data:
        time_data = group['time'].values
        cp_data = group['CP'].values
        response_data = group['RESPONSE'].values

        individual_patient_data[patient_id] = {
            'time': time_data,
            'CP': cp_data,
            'RESPONSE': response_data
        }


    pk_pd_dataframe = pd.read_csv(covariate_file)
    cp_response_data = pk_pd_dataframe[['ID', 'WT', 'AGE', 'SEX']]
    grouped_data = cp_response_data.groupby('ID')
    individual_patient_covariate_data = {}
    for patient_id, group in grouped_data:
        weight_data = group['WT'].values
        age_data = group['AGE'].values
        sex_data = group['SEX'].values

        individual_patient_covariate_data[patient_id] = {
            'WT': weight_data,
            'AGE': age_data,
            'SEX': sex_data
        }

    length = len(individual_patient_data[1]['time'])

    all_patient_data = np.zeros((100, 7, length))

    for i in range(1, 101):
        all_patient_data[i - 1, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 1, 1, :] = individual_patient_data[i]['RESPONSE']
        all_patient_data[i - 1, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 1, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 1, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 1, 5, :] = dose_size
        all_patient_data[i - 1, 6, :] = dose_regime_index

    samples = np.zeros(35, dtype='int')
    
    i = 1
    samples[i] = samples[i - 1] + 46
    samples[i + 1] = samples[i] + 195
    i += 2
    while i < 35:
        samples[i] = samples[i - 1] + 45
        samples[i + 1] = samples[i] + 195
        i += 2
    samples[-1] = samples[-1] - 10
    all_patient_data = all_patient_data[:, :, samples]

    full_time_tensor = torch.tensor(all_patient_data[0, 0, :], dtype=torch.float32)
    return full_time_tensor, all_patient_data

def loss_func(x_hat, labels, weights):
    loss_weights = torch.tensor(weights, dtype=torch.float32)
    loss = torch.abs(x_hat - labels) * loss_weights
    loss = torch.mean(loss)
    return loss

dose_size_1 = 75

dose_sizes = np.array([100, 50, 0])
mean = np.mean(dose_sizes)
std = np.std(dose_sizes)

dose_size_1  = (dose_size_1 - mean) / std

full_time_tensor, val_patient_data_0 = load_validation_data('Num Dose Data/PKPD_75_24_0.csv', 'Num Dose Data/75_24_0.csv', dose_size_1, 0)
full_time_tensor, val_patient_data_1 = load_validation_data('Num Dose Data/PKPD_75_24_1.csv', 'Num Dose Data/75_24_1.csv', dose_size_1, 1)
full_time_tensor, val_patient_data_2 = load_validation_data('Num Dose Data/PKPD_75_24_2.csv', 'Num Dose Data/75_24_2.csv', dose_size_1, 2)
full_time_tensor, val_patient_data_3 = load_validation_data('Num Dose Data/PKPD_75_24_3.csv', 'Num Dose Data/75_24_3.csv', dose_size_1, 3)
full_time_tensor, val_patient_data_4 = load_validation_data('Num Dose Data/PKPD_75_24_4.csv', 'Num Dose Data/75_24_4.csv', dose_size_1, 4)
full_time_tensor, val_patient_data_5 = load_validation_data('Num Dose Data/PKPD_75_24_5.csv', 'Num Dose Data/75_24_5.csv', dose_size_1, 5)
full_time_tensor, val_patient_data_6 = load_validation_data('Num Dose Data/PKPD_75_24_6.csv', 'Num Dose Data/75_24_6.csv', dose_size_1, 6)
full_time_tensor, val_patient_data_7 = load_validation_data('Num Dose Data/PKPD_75_24_7.csv', 'Num Dose Data/75_24_7.csv', dose_size_1, 7)
full_time_tensor, val_patient_data_8 = load_validation_data('Num Dose Data/PKPD_75_24_8.csv', 'Num Dose Data/75_24_8.csv', dose_size_1, 8)
full_time_tensor, val_patient_data_9 = load_validation_data('Num Dose Data/PKPD_75_24_9.csv', 'Num Dose Data/75_24_9.csv', dose_size_1, 9)
full_time_tensor, val_patient_data_10 = load_validation_data('Num Dose Data/PKPD_75_24_10.csv', 'Num Dose Data/75_24_10.csv', dose_size_1, 10)
full_time_tensor, val_patient_data_11 = load_validation_data('Num Dose Data/PKPD_75_24_11.csv', 'Num Dose Data/75_24_11.csv', dose_size_1, 11)
full_time_tensor, val_patient_data_12 = load_validation_data('Num Dose Data/PKPD_75_24_12.csv', 'Num Dose Data/75_24_12.csv', dose_size_1, 12)
full_time_tensor, val_patient_data_13 = load_validation_data('Num Dose Data/PKPD_75_24_13.csv', 'Num Dose Data/75_24_13.csv', dose_size_1, 13)
full_time_tensor, val_patient_data_14 = load_validation_data('Num Dose Data/PKPD_75_24_14.csv', 'Num Dose Data/75_24_14.csv', dose_size_1, 14)
full_time_tensor, val_patient_data_15 = load_validation_data('Num Dose Data/PKPD_75_24_15.csv', 'Num Dose Data/75_24_15.csv', dose_size_1, 15)


val_batch_size = 100
val_patient_dataset_0 = PatientDataset(val_patient_data_0)
val_patient_dataset_1 = PatientDataset(val_patient_data_1)
val_patient_dataset_2 = PatientDataset(val_patient_data_2)
val_patient_dataset_3 = PatientDataset(val_patient_data_3)
val_patient_dataset_4 = PatientDataset(val_patient_data_4)
val_patient_dataset_5 = PatientDataset(val_patient_data_5)
val_patient_dataset_6 = PatientDataset(val_patient_data_6)
val_patient_dataset_7 = PatientDataset(val_patient_data_7)
val_patient_dataset_8 = PatientDataset(val_patient_data_8)
val_patient_dataset_9 = PatientDataset(val_patient_data_9)
val_patient_dataset_10 = PatientDataset(val_patient_data_10)
val_patient_dataset_11 = PatientDataset(val_patient_data_11)
val_patient_dataset_12 = PatientDataset(val_patient_data_12)
val_patient_dataset_13 = PatientDataset(val_patient_data_13)
val_patient_dataset_14 = PatientDataset(val_patient_data_14)
val_patient_dataset_15 = PatientDataset(val_patient_data_15)
val_dataloader_0 = DataLoader(val_patient_dataset_0, batch_size=val_batch_size, shuffle=False)
val_dataloader_1 = DataLoader(val_patient_dataset_1, batch_size=val_batch_size, shuffle=False)
val_dataloader_2 = DataLoader(val_patient_dataset_2, batch_size=val_batch_size, shuffle=False)
val_dataloader_3 = DataLoader(val_patient_dataset_3, batch_size=val_batch_size, shuffle=False)
val_dataloader_4 = DataLoader(val_patient_dataset_4, batch_size=val_batch_size, shuffle=False)
val_dataloader_5 = DataLoader(val_patient_dataset_5, batch_size=val_batch_size, shuffle=False)
val_dataloader_6 = DataLoader(val_patient_dataset_6, batch_size=val_batch_size, shuffle=False)
val_dataloader_7 = DataLoader(val_patient_dataset_7, batch_size=val_batch_size, shuffle=False)
val_dataloader_8 = DataLoader(val_patient_dataset_8, batch_size=val_batch_size, shuffle=False)
val_dataloader_9 = DataLoader(val_patient_dataset_9, batch_size=val_batch_size, shuffle=False)
val_dataloader_10 = DataLoader(val_patient_dataset_10, batch_size=val_batch_size, shuffle=False)
val_dataloader_11 = DataLoader(val_patient_dataset_11, batch_size=val_batch_size, shuffle=False)
val_dataloader_12 = DataLoader(val_patient_dataset_12, batch_size=val_batch_size, shuffle=False)
val_dataloader_13 = DataLoader(val_patient_dataset_13, batch_size=val_batch_size, shuffle=False)
val_dataloader_14 = DataLoader(val_patient_dataset_14, batch_size=val_batch_size, shuffle=False)
val_dataloader_15 = DataLoader(val_patient_dataset_15, batch_size=val_batch_size, shuffle=False)

val_data_loaders = [
                    val_dataloader_1,
                    val_dataloader_2,
                    val_dataloader_3,
                    val_dataloader_4,
                    val_dataloader_5,
                    val_dataloader_6,
                    val_dataloader_7,
                    val_dataloader_8,
                    val_dataloader_9,
                    val_dataloader_10,
                    val_dataloader_11,
                    val_dataloader_12,
                    val_dataloader_13,
                    val_dataloader_14,
                    val_dataloader_15]

dose_times = 4 * np.zeros(15)
for i in range(1, 15):
    dose_times[i] = dose_times[i - 1] + 24

dose_times = torch.tensor(dose_times, dtype=torch.float32)

bottleneck = 16
cost_func = torch.nn.MSELoss()

seeds = [0, 1, 3, 4, 5]
for seed in seeds:
    mse_list = []
    node = NeuralODE(ODEFunc(bottleneck))
    latent_ODE = LatentODE(3, node, bottleneck)
    latent_ODE.load_state_dict(torch.load(f'Saved Weights/node_backprop_solver_pd_{seed}', weights_only=True))
    for data_loader in val_data_loaders:
        for batch in data_loader:
            covariate_input = batch[:, 2:6, 0]
            covariate_input = covariate_input.to(torch.float32)
            data_tensor = batch[:, 1, :].to(torch.float32)
            if batch[0, 6, 0] == 0:
                dose_time_input = None
            if batch[0, 6, 0] == 1:
                dose_time_input = dose_times[:1].expand((batch.size(0), len(dose_times[:1]))).reshape((batch.size(0), len(dose_times[:1])))
            elif batch[0, 6, 0] == 2:
                dose_time_input = dose_times[:2].expand((batch.size(0), len(dose_times[:2]))).reshape((batch.size(0), len(dose_times[:2])))
            elif batch[0, 6, 0] == 3:
                dose_time_input = dose_times[:3].expand((batch.size(0), len(dose_times[:3]))).reshape((batch.size(0), len(dose_times[:3])))
            elif batch[0, 6, 0] == 4:
                dose_time_input = dose_times[:4].expand((batch.size(0), len(dose_times[:4]))).reshape((batch.size(0), len(dose_times[:4])))
            elif batch[0, 6, 0] == 5:
                dose_time_input = dose_times[:5].expand((batch.size(0), len(dose_times[:5]))).reshape((batch.size(0), len(dose_times[:5])))
            elif batch[0, 6, 0] == 6:
                dose_time_input = dose_times[:6].expand((batch.size(0), len(dose_times[:6]))).reshape((batch.size(0), len(dose_times[:6])))
            elif batch[0, 6, 0] == 7:
                dose_time_input = dose_times[:7].expand((batch.size(0), len(dose_times[:7]))).reshape((batch.size(0), len(dose_times[:7])))
            elif batch[0, 6, 0] == 8:
                dose_time_input = dose_times[:8].expand((batch.size(0), len(dose_times[:8]))).reshape((batch.size(0), len(dose_times[:8])))
            elif batch[0, 6, 0] == 9:
                dose_time_input = dose_times[:9].expand((batch.size(0), len(dose_times[:9]))).reshape((batch.size(0), len(dose_times[:9])))
            elif batch[0, 6, 0] == 10:
                dose_time_input = dose_times[:10].expand((batch.size(0), len(dose_times[:10]))).reshape((batch.size(0), len(dose_times[:10])))
            elif batch[0, 6, 0] == 11:
                dose_time_input = dose_times[:11].expand((batch.size(0), len(dose_times[:11]))).reshape((batch.size(0), len(dose_times[:11])))
            elif batch[0, 6, 0] == 12:
                dose_time_input = dose_times[:12].expand((batch.size(0), len(dose_times[:12]))).reshape((batch.size(0), len(dose_times[:12])))
            elif batch[0, 6, 0] == 13:
                dose_time_input = dose_times[:13].expand((batch.size(0), len(dose_times[:13]))).reshape((batch.size(0), len(dose_times[:13])))
            elif batch[0, 6, 0] == 14:
                dose_time_input = dose_times[:14].expand((batch.size(0), len(dose_times[:14]))).reshape((batch.size(0), len(dose_times[:14])))
            elif batch[0, 6, 0] == 15:
                dose_time_input = dose_times[:15].expand((batch.size(0), len(dose_times[:15]))).reshape((batch.size(0), len(dose_times[:15])))
            x_hat = latent_ODE(covariate_input, dose_time_input, full_time_tensor)
            x_hat = x_hat.transpose(1,0)
            x_hat = x_hat.detach().squeeze()
            loss = cost_func(x_hat, data_tensor)
            mse_list.append(loss)
                
    concentration_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    np.save(f'Evaluation Results/node_pd_num_dose_{seed}',np.array(mse_list))
