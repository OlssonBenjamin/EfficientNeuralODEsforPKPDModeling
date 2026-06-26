
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

class SimpleNetwork(nn.Module):
    def __init__(self, num_variables, bottleneck):
       super(SimpleNetwork, self).__init__()
       self.time_func = nn.Sequential(
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
       self.patient_encoding = nn.Sequential(
           nn.Linear(num_variables, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )
    def forward(self, input, dosing_times, predict_times):
        batch_size = input.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32) + 50
        final_time_series = torch.zeros((len(predict_times), batch_size))
        final_time_series[0, :] += 50
        steering_variables = self.patient_encoding(input[:, :-1])
        for i, times in enumerate(predict_times):
            if i == 0:
                continue
            else:
                times = times.unsqueeze(0)
                t_expanded = times.unsqueeze(0).expand((y0.size(0), times.size(0)))
                mask = (dosing_times <= t_expanded)
                masked_dosing_times = dosing_times * mask 
                last_dose = times - masked_dosing_times.max(dim=1, keepdim=True)[0]
                dose_input = torch.cat([last_dose, input[:, -1].unsqueeze(-1)], dim=1)
                dose_encoding = self.dose_encoding(dose_input)
                current_concentration = final_time_series[i - 1, :].unsqueeze(-1)
                encoding = torch.cat([current_concentration, steering_variables, dose_encoding], dim=1)
                final_time_series[i, :] = self.time_func(encoding).squeeze()
        return final_time_series[:, :]
    

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0)

def load_data(covariate_file, data_file, dose_size, dose_regime_index):
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
    samples[i] = samples[i - 1] + 41
    samples[i + 1] = samples[i] + 200
    i += 2
    while i < 35:
        samples[i] = samples[i - 1] + 40
        samples[i + 1] = samples[i] + 200
        i += 2
    all_patient_data = all_patient_data[:, :, samples]

    full_time_tensor = torch.tensor(all_patient_data[0, 0, :], dtype=torch.float32)
    check_time_tensor = torch.tensor(all_patient_data[0, 0, :], dtype=torch.float32)
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return full_time_tensor, mask, all_patient_data

def loss_func(x_hat, labels, weights):
    loss_weights = torch.tensor(weights, dtype=torch.float32)
    loss = torch.abs(x_hat - labels) * loss_weights
    loss = torch.mean(loss)
    return loss

dose_size_1 = 0
dose_size_2 = 25
dose_size_3 = 50
dose_size_4 = 75
dose_size_5 = 100
dose_size_6 = 125
dose_size_7 = 150
dose_size_8 = 175
dose_size_9 = 200
dose_size_10 = 225
dose_size_11 = 250

dose_sizes = np.array([100, 50, 0])
mean = np.mean(dose_sizes)
std = np.std(dose_sizes)

dose_size_1  = (dose_size_1 - mean) / std
dose_size_2  = (dose_size_2 - mean) / std
dose_size_3  = (dose_size_3 - mean) / std
dose_size_4  = (dose_size_4 - mean) / std
dose_size_5  = (dose_size_5 - mean) / std
dose_size_6  = (dose_size_6 - mean) / std
dose_size_7  = (dose_size_7 - mean) / std
dose_size_8  = (dose_size_8 - mean) / std
dose_size_9  = (dose_size_9 - mean) / std
dose_size_10  = (dose_size_10 - mean) / std
dose_size_11  = (dose_size_11 - mean) / std


model = torch.load('Models/mlp_pd_1')

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)

full_time_tensor, val_mask, val_patient_data_1 = load_data('Dose Size Data/WT_AGE_SEX_0_24_8.csv', 'Dose Size Data/PK_PD_0_24_8.csv', dose_size_1, 0)
full_time_tensor, val_mask, val_patient_data_2 = load_data('Dose Size Data/WT_AGE_SEX_25_24_8.csv', 'Dose Size Data/PK_PD_25_24_8.csv', dose_size_2, 0)
full_time_tensor, val_mask, val_patient_data_3 = load_data('Dose Size Data/WT_AGE_SEX_50_24_8.csv', 'Dose Size Data/PK_PD_50_24_8.csv', dose_size_3, 0)
full_time_tensor, val_mask, val_patient_data_4 = load_data('Dose Size Data/WT_AGE_SEX_75_24_8.csv', 'Dose Size Data/PK_PD_75_24_8.csv', dose_size_4, 0)
full_time_tensor, val_mask, val_patient_data_5 = load_data('Dose Size Data/WT_AGE_SEX_100_24_8.csv', 'Dose Size Data/PK_PD_100_24_8.csv', dose_size_5, 0)
full_time_tensor, val_mask, val_patient_data_6 = load_data('Dose Size Data/WT_AGE_SEX_125_24_8.csv', 'Dose Size Data/PK_PD_125_24_8.csv', dose_size_6, 0)
full_time_tensor, val_mask, val_patient_data_7 = load_data('Dose Size Data/WT_AGE_SEX_150_24_8.csv', 'Dose Size Data/PK_PD_150_24_8.csv', dose_size_7, 0)
full_time_tensor, val_mask, val_patient_data_8 = load_data('Dose Size Data/WT_AGE_SEX_175_24_8.csv', 'Dose Size Data/PK_PD_175_24_8.csv', dose_size_8, 0)
full_time_tensor, val_mask, val_patient_data_9 = load_data('Dose Size Data/WT_AGE_SEX_200_24_8.csv', 'Dose Size Data/PK_PD_200_24_8.csv', dose_size_9, 0)
full_time_tensor, val_mask, val_patient_data_10 = load_data('Dose Size Data/WT_AGE_SEX_225_24_8.csv', 'Dose Size Data/PK_PD_225_24_8.csv', dose_size_10, 0)
full_time_tensor, val_mask, val_patient_data_11 = load_data('Dose Size Data/WT_AGE_SEX_250_24_8.csv', 'Dose Size Data/PK_PD_250_24_8.csv', dose_size_11, 0)


val_batch_size = 100
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

val_data_loaders = [val_dataloader_1,
                    val_dataloader_2,
                    val_dataloader_3,
                    val_dataloader_4,
                    val_dataloader_5,
                    val_dataloader_6,
                    val_dataloader_7,
                    val_dataloader_8,
                    val_dataloader_9,
                    val_dataloader_10,
                    val_dataloader_11]

dose_times_1 = 4 * np.zeros(8)


for i in range(1, 8):
    dose_times_1[i] = dose_times_1[i - 1] + 24


dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)

mse_list = []

cost_func = torch.nn.MSELoss()
for data_loader in val_data_loaders:
    for batch in data_loader:
        covariate_input = batch[:, 2:6, 0]
        covariate_input = covariate_input.to(torch.float32)
        data_tensor = batch[:, 1, :].to(torch.float32)
        if batch[0, 6, 0] == 0:
            dose_time_input = dose_times_1.expand((batch.size(0), len(dose_times_1))).reshape((batch.size(0), len(dose_times_1)))
        x_hat = model(covariate_input, dose_time_input, full_time_tensor)
        x_hat = x_hat[val_mask].transpose(1,0)
        x_hat = x_hat.detach().squeeze()
        loss = cost_func(x_hat, data_tensor)
        mse_list.append(loss)
            
concentration_list = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
np.save('mlp_dose_size_pk_1',np.array(mse_list))
