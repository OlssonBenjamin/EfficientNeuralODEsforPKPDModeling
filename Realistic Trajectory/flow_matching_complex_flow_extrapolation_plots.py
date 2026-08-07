import torchdiffeq as tdq
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import copy

class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class NeuralODE(nn.Module):
   def __init__(self, ode_func):
       super(NeuralODE, self).__init__()
       self.ode_func = ode_func
       self.step_size = 0.1

   def forward(self, input, t):
       return odeint(self.ode_func, input, t, method='euler', options=dict(step_size=self.step_size))

class TrainODEFunc(nn.Module):
   def __init__(self, bottleneck):
       super(TrainODEFunc, self).__init__()
       self.increasing_net = nn.Sequential(
           nn.Linear(bottleneck + 2, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
       self.decreasing_net = nn.Sequential(
           nn.Linear(bottleneck, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
       self.bottleneck = bottleneck
       self.peak_dose = 4

   def forward(self, t, y):
        current_concentration = y[:, 0].unsqueeze(1)
        steering_variables = y[:, 1:self.bottleneck + 1]
        dosing_times = y[:, self.bottleneck + 1:-1]
        dose_size = y[:, -1].unsqueeze(-1)
        t_expanded = t.unsqueeze(0).expand((current_concentration.size(0), dosing_times.size(1)))
        mask = (dosing_times <= t_expanded)
        num_dose = torch.min(torch.tensor([torch.sum(mask[0]), 2]))
        num_dose = num_dose.expand([y.size(0), 1]).to(torch.float32)
        masked_dosing_times = dosing_times * mask 
        last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]
        if (last_dose >= self.peak_dose).any():
            ode_input = torch.cat([steering_variables], dim=1)
            dzdt = -(self.decreasing_net(ode_input)) * current_concentration * min(last_dose[0] - self.peak_dose, 1)
        else:
            ode_input = torch.cat([steering_variables, dose_size, last_dose], dim=1)
            dzdt = self.increasing_net(ode_input)
        dzdt = torch.cat([dzdt, torch.zeros_like(steering_variables), torch.zeros_like(dosing_times), torch.zeros_like(dose_size)], dim=-1)
        return dzdt




   
class TrainODE(nn.Module):
    def __init__(self, num_variables, bottleneck, zero_dose):
       super(TrainODE, self).__init__()
       self.patient_encoding = nn.Sequential(
           nn.Linear(num_variables, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )
       self.ode = TrainODEFunc(bottleneck)
       self.node = NeuralODE(self.ode)
       self.std = 10
       self.zero_dose = zero_dose
       self.peak_dose = 4

    def forward(self, input, dosing_times, predict_times, trajectories):
        x = self.patient_encoding(input[:, :-1])
        fine_predict_times = torch.linspace(predict_times[0], predict_times[-1], steps=int(predict_times[-1] * 10 + 1))
        mask = ~torch.isin(fine_predict_times, predict_times[-1])
        fine_predict_times = fine_predict_times[mask]
        num_samples = 1
        random_indices = torch.randint(low=0, high=len(fine_predict_times), size=(num_samples,))
        random_times = fine_predict_times[random_indices] 
        i = torch.searchsorted(predict_times, random_times)

        in_mask = torch.isin(random_times, predict_times) 
        smaller_idx = i - 1
        bigger_idx = i

        smaller_idx[in_mask] = i[in_mask]
        bigger_idx[in_mask] = i[in_mask] + 1


        smaller_times = predict_times[smaller_idx]
        bigger_times = predict_times[bigger_idx]

        batch_size = trajectories.size(0)
        sm_idx_expand = smaller_idx.unsqueeze(0).expand(batch_size, -1)  
        bg_idx_expand = bigger_idx.unsqueeze(0).expand(batch_size, -1)  
        smaller_values = torch.gather(trajectories, 1, sm_idx_expand)   
        bigger_values = torch.gather(trajectories, 1, bg_idx_expand)    
  
        alpha = (random_times - smaller_times) / (bigger_times - smaller_times)
        alpha = alpha.unsqueeze(0).expand(batch_size, -1)  

        t_expanded = random_times[0].unsqueeze(0).expand((input.size(0), dosing_times.size(1)))
        mask = (dosing_times <= t_expanded)
        masked_dosing_times = dosing_times * mask 
        last_dose = random_times[0] - masked_dosing_times.max(dim=1, keepdim=True)[0] 

        if last_dose[0] < self.peak_dose:        
            interpolated_values = smaller_values + alpha * (bigger_values - smaller_values)
        
        else:
            interpolated_values = smaller_values**(1-alpha) * bigger_values**alpha

        interpolants = torch.normal(interpolated_values, self.std)

        interp_i = interpolants[:, 0].unsqueeze(1)

        encoding_i = torch.cat([
                interp_i,
                x,
                dosing_times,
                input[:, -1].unsqueeze(-1)
            ], dim=1)
            # Evaluate ODE at time = rt
        rt = random_times[0]
        dzdt = self.ode(rt, encoding_i)[:, 0]  # shape: [batch_size]

        return dzdt, random_times, bigger_idx, smaller_idx, bigger_times, smaller_times, interpolants, last_dose
    
    
    def solve(self, input, dosing_times, predict_times):
        x = self.patient_encoding(input[:, :-1])
        batch_size = x.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        final_time_series = nn.functional.relu(latent_time_series[:, :, 0])
        return final_time_series

    
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

    all_patient_data = np.zeros((75, 8, length))

    for i in range(1, 76):
        all_patient_data[i - 1, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 1, 1, :] = individual_patient_data[i]['RESPONSE']
        all_patient_data[i - 1, 2, :] = individual_patient_data[i]['CP']
        all_patient_data[i - 1, 3, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 1, 4, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 1, 5, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 1, 6, :] = dose_size
        all_patient_data[i - 1, 7, :] = dose_regime_index

    samples = np.zeros(35, dtype='int')
        
    i = 1
    samples[i] = samples[i - 1] + 41
    samples[i + 1] = samples[i] + 200
    i += 2
    while i < 35:
        samples[i] = samples[i - 1] + 40
        samples[i + 1] = samples[i] + 200
        i += 2
    samples[-1] = samples[-1] - 10
    sampled_patient_data = all_patient_data[:, :, samples]

    full_time_tensor = np.unique(sampled_patient_data[:, 0, :].flatten())
    rounded_tensor = np.round(full_time_tensor, decimals=1)
    rounded_tensor = torch.tensor(full_time_tensor, dtype=torch.float64)


    full_time_tensor = np.unique(sampled_patient_data[:, 0, :].flatten())
    return rounded_tensor, sampled_patient_data

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

bottleneck = 4

seed = 0
model = torch.load(f'flow_matching_model_complex_flow_{seed}', weights_only=False)

full_time_tensor, val_patient_data_1 = load_data('Dose Size Data/0_24_8_100.csv', 'Dose Size Data/PK_PD_0_24_8_100.csv', dose_size_1, 0)
full_time_tensor, val_patient_data_2 = load_data('Dose Size Data/25_24_8_100.csv', 'Dose Size Data/PK_PD_25_24_8_100.csv', dose_size_2, 0)
full_time_tensor, val_patient_data_3 = load_data('Dose Size Data/50_24_8_100.csv', 'Dose Size Data/PK_PD_50_24_8_100.csv', dose_size_3, 0)
full_time_tensor, val_patient_data_4 = load_data('Dose Size Data/75_24_8_100.csv', 'Dose Size Data/PK_PD_75_24_8_100.csv', dose_size_4, 0)
full_time_tensor, val_patient_data_5 = load_data('Dose Size Data/100_24_8_100.csv', 'Dose Size Data/PK_PD_100_24_8_100.csv', dose_size_5, 0)
full_time_tensor, val_patient_data_6 = load_data('Dose Size Data/125_24_8_100.csv', 'Dose Size Data/PK_PD_125_24_8_100.csv', dose_size_6, 0)
full_time_tensor, val_patient_data_7 = load_data('Dose Size Data/150_24_8_100.csv', 'Dose Size Data/PK_PD_150_24_8_100.csv', dose_size_7, 0)
full_time_tensor, val_patient_data_8 = load_data('Dose Size Data/175_24_8_100.csv', 'Dose Size Data/PK_PD_175_24_8_100.csv', dose_size_8, 0)
full_time_tensor, val_patient_data_9 = load_data('Dose Size Data/200_24_8_100.csv', 'Dose Size Data/PK_PD_200_24_8_100.csv', dose_size_9, 0)
full_time_tensor, val_patient_data_10 = load_data('Dose Size Data/225_24_8_100.csv', 'Dose Size Data/PK_PD_225_24_8_100.csv', dose_size_10, 0)
full_time_tensor, val_patient_data_11 = load_data('Dose Size Data/250_24_8_100.csv', 'Dose Size Data/PK_PD_250_24_8_100.csv', dose_size_11, 0)


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

val_data_loaders = [
                    val_dataloader_8]

dose_times_1 = 4 * np.zeros(9)


for i in range(1, 9):
    dose_times_1[i] = dose_times_1[i - 1] + 24


dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)

smooth_time_tensor = torch.linspace(full_time_tensor[0], full_time_tensor[-1], steps = int((full_time_tensor[-1] - full_time_tensor[0])))

cost_func = torch.nn.MSELoss()
i = 0
for val_dataloader in val_data_loaders:
            for batch in val_dataloader:
                    covariate_input = batch[:, 3:7, 0]
                    covariate_input = covariate_input.to(torch.float32)
                    data_tensor = batch[:, 2, :].to(torch.float32)
                    dose_time_input = dose_times_1.expand((batch.size(0), len(dose_times_1)))
                    x_hat = model.solve(covariate_input, dose_time_input, smooth_time_tensor).transpose(0, 1).squeeze()
                    for z in range(24, 25):
                        fig, ax = plt.subplots(figsize=(10, 5))

                        # --- Style ---
                        fig.patch.set_facecolor('#ffffff')
                        ax.set_facecolor('#ffffff')

                        # Grid
                        ax.grid(color='#000000', linestyle='--', linewidth=0.4, alpha=0.15)
                        ax.set_axisbelow(True)

                        # Prediction line (smooth)
                        ax.plot(
                            smooth_time_tensor,
                            x_hat.detach()[z, :],
                            color='#1a6fcc',
                            linewidth=2,
                            alpha=0.95,
                            label='Prediction $\\hat{x}$',
                            zorder=3
                        )

                        # Training data scatter
                        ax.scatter(
                            full_time_tensor,
                            data_tensor.squeeze()[z, :],
                            color='#e03c3c',
                            s=30,
                            alpha=0.85,
                            zorder=4,
                            label='Validation Data\n(Dose size of 175 mg)',
                            edgecolors='#c02020',
                            linewidths=0.5
                        )

                        # --- Labels & Title ---
                        ax.set_title(f'Flow Matching Neural ODE with Realistic Trajectory', color='#111111', fontsize=14, fontweight='bold', pad=14)
                        ax.set_xlabel('Time', color='#333333', fontsize=11)
                        ax.set_ylabel('Concentration (mg/L)', color='#333333', fontsize=11)

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
                        plt.savefig(f'Figures/Realistic_Trajectory.pdf',  bbox_inches='tight', facecolor=fig.get_facecolor()) 

        