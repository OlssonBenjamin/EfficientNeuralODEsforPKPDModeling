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

class Encoder(nn.Module):
    def __init__(self, hidden_layer, bottleneck):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(4, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, 4 * hidden_layer),
            nn.Linear(4 * hidden_layer,  bottleneck)
        )
        self.bottleneck = bottleneck

    def forward(self, input):
        x = self.encoder(input)
        return x
    
class ODEFunc(nn.Module):
   def __init__(self, bottleneck):
       super(ODEFunc, self).__init__()
       self.net = nn.Sequential(
           nn.Linear(bottleneck * 2 + 1, 128),
           nn.SiLU(),
           nn.Linear(128, 512),
           nn.SiLU(),
           nn.Linear(512, 1)
       )
       self.dose_encoding = nn.Sequential(
           nn.Linear(2, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )

   def forward(self, t, y):
        y0 = y[:, 0].unsqueeze(1)
        steering_variables = y[:, 1:17]
        dosing_times = y[:, 17:]
        t_expanded = t.unsqueeze(0).expand((y0.size(0), dosing_times.size(1)))
        mask = (dosing_times <= t_expanded)
        masked_dosing_times = dosing_times * mask 
        last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]
        total_doses = mask.sum(dim=1, keepdim=True).float()
        dose_input = torch.cat([last_dose, total_doses], dim=1)
        dose_encoding = self.dose_encoding(dose_input)
        ode_input = torch.cat([y0, steering_variables, dose_encoding], dim=1)
        dzdt = self.net(ode_input)
        dzdt = torch.cat([dzdt, torch.zeros_like(steering_variables), torch.zeros_like(dosing_times)], dim=1)
        return dzdt

class NeuralODE(nn.Module):
   def __init__(self, ode_func):
       super(NeuralODE, self).__init__()
       self.ode_func = ode_func

   def forward(self, input, t):
       return odeint(self.ode_func, input, t, method='euler')

class LatentODE(nn.Module):
    def __init__(self, encoder, node):
       super(LatentODE, self).__init__()
       self.encoder = encoder
       self.node = node

    def forward(self, input, dosing_times, predict_times):
        x = self.encoder(input)
        batch_size = x.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32)
        encoding = torch.cat([y0, x, dosing_times], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        return latent_time_series[:, :, 0]

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0)

def load_data(data_file, covariate_file, dose_size, dose_regime_index):
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

    all_patient_data = np.zeros((75, 7, length))

    for i in range(1, 76):
        all_patient_data[i - 1, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 1, 1, :] = individual_patient_data[i]['CP']
        all_patient_data[i - 1, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 1, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 1, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 1, 5, :] = dose_size
        all_patient_data[i - 1, 6, :] = dose_regime_index

    #samples = np.zeros(29, dtype='int')
    weights = np.ones(35)
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
    fine_time_tensor = torch.linspace(all_patient_data[0, 0, 0], all_patient_data[0, 0, -1], 408)
    full_time_tensor = torch.cat([check_time_tensor, fine_time_tensor]).sort()[0].unique()
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return full_time_tensor, mask, all_patient_data

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

    all_patient_data = np.zeros((25, 7, length))

    for i in range(76, 101):
        all_patient_data[i - 76, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 76, 1, :] = individual_patient_data[i]['CP']
        all_patient_data[i - 76, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 76, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 76, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 76, 5, :] = dose_size
        all_patient_data[i - 76, 6, :] = dose_regime_index

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
    fine_time_tensor = torch.linspace(all_patient_data[0, 0, 0], all_patient_data[0, 0, -1], 408)
    full_time_tensor = torch.cat([check_time_tensor, fine_time_tensor]).sort()[0].unique()
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return full_time_tensor, mask, all_patient_data


dose_size_1 = 100
dose_size_2 = 50
dose_size_3 = 0

num_doses_1 = 5
num_doses_2 = 10

full_time_tensor, mask, all_patient_data_1 = load_data('Zipped_Backups/Random Data/PKPD_100_24_5_Random.csv',
                                                       'Zipped_Backups/Random Data/100_24_5_Random.csv',
                                                       dose_size_1, 0)
_, _, all_patient_data_2 = load_data('Zipped_Backups/Random Data/PKPD_50_24_10_Random.csv',
                                     'Zipped_Backups/Random Data/50_24_10_Random.csv',
                                     dose_size_2, 1)
_, _, all_patient_data_3 = load_data('Zipped_Backups/Random Data/PKPD_0_24_10_Random.csv',
                                     'Zipped_Backups/Random Data/0_24_10_Random.csv',
                                     dose_size_3, 2)

val_time_tensor_1, val_mask_1, val_patient_data_1 = load_validation_data('Zipped_Backups/Random Data/PKPD_100_24_5_Random.csv',
                                                       'Zipped_Backups/Random Data/100_24_5_Random.csv',
    dose_size_1, 0)
val_time_tensor_2, val_mask_2, val_patient_data_2 = load_validation_data('Zipped_Backups/Random Data/PKPD_50_24_10_Random.csv',
                                     'Zipped_Backups/Random Data/50_24_10_Random.csv',
    dose_size_2, 1)
val_time_tensor_3, val_mask_3, val_patient_data_3 = load_validation_data('Zipped_Backups/Random Data/PKPD_0_24_10_Random.csv',
                                     'Zipped_Backups/Random Data/0_24_10_Random.csv',
    dose_size_3, 2)
dose_times_1 = np.zeros(5)
dose_times_2 = np.zeros(10)

for i in range(1, 5):
    dose_times_1[i] = dose_times_1[i - 1] + 24
    dose_times_2[i] = dose_times_2[i - 1] + 24

for i in range(5, 10):
    dose_times_2[i] = dose_times_2[i - 1] + 24

dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)
dose_times_2 = torch.tensor(dose_times_2, dtype=torch.float32)


batch_size = 16
patient_dataset_1 = PatientDataset(all_patient_data_1)
patient_dataset_2 = PatientDataset(all_patient_data_2)
patient_dataset_3 = PatientDataset(all_patient_data_3)
dataloader_1 = DataLoader(patient_dataset_1, batch_size=batch_size, shuffle=True)
dataloader_2 = DataLoader(patient_dataset_2, batch_size=batch_size, shuffle=True)
dataloader_3 = DataLoader(patient_dataset_3, batch_size=batch_size, shuffle=True)

val_batch_size = 25
val_patient_dataset_1 = PatientDataset(val_patient_data_1)
val_patient_dataset_2 = PatientDataset(val_patient_data_2)
val_patient_dataset_3 = PatientDataset(val_patient_data_3)
val_dataloader_1 = DataLoader(val_patient_dataset_1, batch_size=val_batch_size, shuffle=False)
val_dataloader_2 = DataLoader(val_patient_dataset_2, batch_size=val_batch_size, shuffle=False)
val_dataloader_3 = DataLoader(val_patient_dataset_3, batch_size=val_batch_size, shuffle=False)

train_dataloaders = [dataloader_1, dataloader_2, dataloader_3]
val_data_loaders = [val_dataloader_1, val_dataloader_2, val_dataloader_3]

bottleneck = 16
encoder = Encoder(32, bottleneck)
ode_func = ODEFunc(bottleneck)
node = NeuralODE(ode_func)
latent_ODE = LatentODE(encoder, node)
latent_ODE.apply(initialize_weights)
epochs = 300
optimizer = torch.optim.Adam(latent_ODE.parameters(), lr=0.001)
best_loss = np.inf
cost_func = torch.nn.MSELoss()

for epoch in range(epochs):
    random.shuffle(train_dataloaders)
    for dataloader in train_dataloaders:
        for batch in dataloader:
            total_loss = 0
            optimizer.zero_grad()
            covariate_input = batch[:, 2:6, 0]
            covariate_input = covariate_input.to(torch.float32)
            data_tensor = batch[:, 1, :].to(torch.float32)
            if batch[0, 6, 0] == 0:
                dose_time_input = dose_times_1.expand((batch.size(0), len(dose_times_1))).reshape((batch.size(0), len(dose_times_1)))
            elif batch[0, 6, 0] == 1:
                dose_time_input = dose_times_2.expand((batch.size(0), len(dose_times_2))).reshape((batch.size(0), len(dose_times_2)))
            else:
                dose_times_3 = np.random.choice(len(dose_times_2)) + 1
                dose_times_3 = dose_times_2[:dose_times_3]
                dose_time_input = dose_times_3.expand((batch.size(0), len(dose_times_3))).reshape((batch.size(0), len(dose_times_3)))
            x_hat = latent_ODE(covariate_input, dose_time_input, full_time_tensor)
            x_hat = x_hat[mask].transpose(1,0)
            loss = cost_func(x_hat, data_tensor)
            if torch.any(loss.isnan()):
                loss = torch.tensor(100, dtype=torch.float32)
                loss.requires_grad = True
            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_ODE.parameters(), max_norm=1.0)
            optimizer.step()
    with torch.no_grad():
        total_loss = 0
        for val_dataloader in val_data_loaders:
            for batch in val_dataloader:
                covariate_input = batch[:, 2:6, 0]
                covariate_input = covariate_input.to(torch.float32)
                data_tensor = batch[:, 1, :].to(torch.float32)
                if batch[0, 6, 0] == 0:
                    dose_time_input = dose_times_1.expand((batch.size(0), len(dose_times_1))).reshape((batch.size(0), len(dose_times_1)))
                elif batch[0, 6, 0] == 1:
                    dose_time_input = dose_times_2.expand((batch.size(0), len(dose_times_2))).reshape((batch.size(0), len(dose_times_2)))
                else:
                    dose_times_3 = np.random.choice(len(dose_times_2)) + 1
                    dose_times_3 = dose_times_2[:dose_times_3]
                    dose_time_input = dose_times_3.expand((batch.size(0), len(dose_times_3))).reshape((batch.size(0), len(dose_times_3)))
                x_hat = latent_ODE(covariate_input, dose_time_input, full_time_tensor)
                x_hat = x_hat[mask].transpose(1,0)
                loss = cost_func(x_hat, data_tensor)
                total_loss += loss / (len(val_data_loaders))
        print(total_loss.item())
    if total_loss.item() < best_loss:
        torch.save(latent_ODE, '1_dimension_euler_batched')
    if total_loss.item() < 0.75:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001 
    if total_loss.item() < best_loss:
        if total_loss < 0.75:
            batch_index = 0
            for val_dataloader in val_data_loaders:
                for batch in val_dataloader:
                    if batch_index == 1:
                        covariate_input = batch[:, 2:6, 0]
                        covariate_input = covariate_input.to(torch.float32)
                        data_tensor = batch[:, 1, :].to(torch.float32)
                        dose_time_input = dose_times_2.expand((batch.size(0), len(dose_times_2)))
                        x_hat = latent_ODE(covariate_input, dose_time_input, full_time_tensor).transpose(0, 1).squeeze()
                        for z in range(25):
                            fig, ax = plt.subplots(figsize=(10, 5))

                            # --- Style ---
                            fig.patch.set_facecolor('#ffffff')
                            ax.set_facecolor('#ffffff')

                            # Grid
                            ax.grid(color='#000000', linestyle='--', linewidth=0.4, alpha=0.15)
                            ax.set_axisbelow(True)

                            # Prediction line (smooth)
                            ax.plot(
                                full_time_tensor,
                                x_hat.detach()[z, :],
                                color='#1a6fcc',
                                linewidth=2,
                                alpha=0.95,
                                label='Prediction $\\hat{x}$',
                                zorder=3
                            )

                            # Subtle fill under the prediction curve
                            ax.fill_between(
                                full_time_tensor,
                                x_hat.detach()[z, :],
                                alpha=0.08,
                                color='#1a6fcc',
                                zorder=2
                            )

                            # Training data scatter
                            ax.scatter(
                                full_time_tensor[mask],
                                data_tensor.squeeze()[z, :],
                                color='#e03c3c',
                                s=30,
                                alpha=0.85,
                                zorder=4,
                                label='Validation Data',
                                edgecolors='#c02020',
                                linewidths=0.5
                            )

                            # --- Labels & Title ---
                            ax.set_title(f'Neural ODE Predictions on Test Data', color='#111111', fontsize=14, fontweight='bold', pad=14)
                            ax.set_xlabel('Time', color='#333333', fontsize=11)
                            ax.set_ylabel('Value', color='#333333', fontsize=11)

                            ax.tick_params(colors='#444444', labelsize=9)
                            for spine in ax.spines.values():
                                spine.set_edgecolor('#cccccc')

                            # --- Legend ---
                            ax.legend(
                                frameon=True,
                                facecolor='#f9f9f9',
                                edgecolor='#cccccc',
                                labelcolor='#222222',
                                fontsize=10,
                                loc='upper right'
                            )

                            plt.tight_layout()
                            plt.savefig(f'Test_Data_Extrapolation_nonfm{z}.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                            plt.show()
                            plt.close()
                batch_index += 1
        best_loss = total_loss.item()

