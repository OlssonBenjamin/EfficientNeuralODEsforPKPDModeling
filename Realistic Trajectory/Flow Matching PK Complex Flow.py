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
import time

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'  

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
        return self.mean_param, self.cov_param
    def sample(self):
        if self.mean_param is None or self.cov_param is None:
            raise ValueError("Call finalize() before sampling.")
        num_params = self.mean_param.shape[0]

        eps = torch.randn(num_params)
        L = torch.linalg.cholesky(self.cov_param)  
        sample_vec = self.mean_param + L @ eps
        return sample_vec
    def get_mean(self):
        sample_vec = self.mean_param 
        return sample_vec


def load_parameters_into_model(model, param_vector):
    """
    Given a 1D param_vector sampled from SWAG (or any other method),
    load it back into the model's parameters in the correct shapes.
    """
    # We'll build a fresh state_dict whose tensor shapes match the original model.
    new_state_dict = {}
    
    # We'll track where we are in 'param_vector'.
    offset = 0
    
    # Iterate over the model's existing parameters in order
    for key, old_tensor in model.state_dict().items():
        # Number of elements in this parameter
        numel = old_tensor.numel()
        
        # Extract that many entries from 'param_vector'
        new_tensor = param_vector[offset : offset + numel].view(old_tensor.shape)
        
        # Store them in our new dict, under the same key
        new_state_dict[key] = new_tensor
        
        # Advance the offset
        offset += numel
    
    # Finally, load our new state_dict into the model
    model.load_state_dict(new_state_dict)

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
            a = (smaller_values - bigger_values) / 16
            b = (bigger_values - smaller_values) / 2        
            interpolated_values = a * last_dose**2 + b * last_dose + smaller_values
        else:
            interpolated_values = smaller_values**(1-alpha) * bigger_values**alpha
            a = 0 
            b = 0

        if (input[0, -1] == self.zero_dose):
                interpolants = torch.normal(interpolated_values, 0.5)
        else:
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

        return dzdt, random_times, bigger_idx, smaller_idx, bigger_times, smaller_times, interpolants, last_dose, a, b, smaller_values, interpolated_values
    

    def decay_std(self):
        if self.std >= 1:
            self.std *= 0.99
    
    def solve(self, input, dosing_times, predict_times):
        x = self.patient_encoding(input[:, :-1])
        batch_size = x.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        final_time_series = nn.functional.relu(latent_time_series[:, :, 0])
        #self.decay_std()
        return final_time_series
        

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

    samples = np.zeros(35, dtype='int')
    
    i = 1
    samples[i] = samples[i - 1] + 40
    samples[i + 1] = samples[i] + 200
    i += 2
    while i < 35:
        samples[i] = samples[i - 1] + 40
        samples[i + 1] = samples[i] + 200
        i += 2
    sample_patient_data = all_patient_data[:, :, samples]
    full_time_tensor = torch.tensor(all_patient_data[0, 0, :], dtype=torch.float32)
    check_time_tensor = torch.tensor(sample_patient_data[0, 0, :], dtype=torch.float32)

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
        all_patient_data[27, 1, :],
        color='#1a6fcc',
        linewidth=2,
        alpha=0.95,
        label='Prediction $\\hat{x}$',
        zorder=3
    )

    """ # Subtle fill under the prediction curve
    ax.fill_between(
        full_time_tensor,
        all_patient_data[27, 1, :],
        alpha=0.08,
        color='#1a6fcc',
        zorder=2
    ) """

    # Training data scatter
    ax.scatter(
        check_time_tensor,
        sample_patient_data[27, 1, :],
        color='#e03c3c',
        s=30,
        alpha=0.85,
        zorder=4,
        label='Training Data Points',
        edgecolors='#c02020',
        linewidths=0.5
    )

    # --- Labels & Title ---
    ax.set_title(f'Training Data Points vs. Full Simulated PK Trajectory', color='#111111', fontsize=14, fontweight='bold', pad=14)
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
    if dose_regime_index == -1:
        plt.savefig(f'Train_Sim_PK.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()
        plt.close()

    full_time_tensor = torch.tensor(sample_patient_data[0, 0, :], dtype=torch.float32)
    check_time_tensor = torch.tensor(sample_patient_data[0, 0, :], dtype=torch.float32)
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return full_time_tensor, mask, sample_patient_data

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
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return full_time_tensor, mask, all_patient_data

dose_size_1 = 100
dose_size_2 = 50
dose_size_3 = 0

dose_sizes = np.array([100, 50, 0])

dose_size_1  = (dose_size_1 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_2  = (dose_size_2 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_3  = (dose_size_3 - np.mean(dose_sizes)) / np.std(dose_sizes)

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


batch_size = 8
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

bottleneck = 4
ode_func = TrainODEFunc(bottleneck)
model = TrainODE(3, bottleneck, dose_size_3)
model.apply(initialize_weights)
epochs = 5001
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
best_loss = np.inf
cost_func = torch.nn.MSELoss()
train_loss_func = torch.nn.MSELoss(reduction='sum')
training_loss = np.zeros(epochs)
validation_loss = np.zeros(epochs)
length_training_data = len(all_patient_data_1) + len(all_patient_data_2) + len(all_patient_data_3)
swag_tracker = SWAGTracker()

start_time = time.time()
for epoch in range(epochs):
    train_loss = 0
    for train_dataloader in train_dataloaders:
        for batch in train_dataloader:
            total_loss = 0
            optimizer.zero_grad()
            covariate_input = batch[:, 2:6, 0]
            covariate_input = covariate_input.to(torch.float32)
            data_tensor = torch.abs(batch[:, 1, :].to(torch.float32))
            if batch[0, 6, 0] == 0:
                dose_time_input = dose_times_1.expand((batch.size(0), len(dose_times_1)))
            elif batch[0, 6, 0] == 1:
                dose_time_input = dose_times_2.expand((batch.size(0), len(dose_times_2)))
            else:
                dose_times_3 = np.random.choice(len(dose_times_2)) + 1
                if dose_times_3 > 0:
                    dose_times_3 = dose_times_2[:dose_times_3]
                    dose_time_input = dose_times_3.expand((batch.size(0), len(dose_times_3)))
                else:
                    dose_time_input = None
            
            dzdt, random_times, bigger_idx, smaller_idx, bigger_times, smaller_times, interpolants, last_dose, a, b, smaller_values, original_interpolants = model(covariate_input, dose_time_input, full_time_tensor, data_tensor)
            if last_dose[0] < 4:
                if batch[0, 6, 0] == 2:
                    conditional_field = interpolants * 0
                else:
                    m = torch.tensor(2 * a * last_dose + b).to(torch.float32)
                    conditional_field = m
            else:
                if batch[0, 6, 0] == 2:
                    conditional_field = -interpolants
                else:
                    conditional_field = torch.tensor((torch.log(data_tensor[:, bigger_idx] / data_tensor[:, smaller_idx]) / (bigger_times - smaller_times) * interpolants * min(last_dose[0] - 4, 1))).to(torch.float32)
            loss = cost_func(dzdt.squeeze(), conditional_field.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss
    num = np.random.uniform()
    if epoch >= 2001 and (num < 0.1):
        swag_tracker.collect_model(model)
    if epoch == 2000 or best_loss < 0.75:
        #model = torch.load('flow_matching_model_swag_exponential_polynomial_11')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    with torch.no_grad():
        total_loss = 0
        if (epoch) % 100 == 0:
            print(f'Epoch: {epoch}')
            print(f'Training Loss: {train_loss.item()}')
            for val_dataloader in val_data_loaders:
                for batch in val_dataloader:
                    covariate_input = batch[:, 2:6, 0]
                    covariate_input = covariate_input.to(torch.float32)
                    data_tensor = batch[:, 1, :].to(torch.float32)
                    if batch[0, 6, 0] == 0:
                        dose_time_input = dose_times_1.expand((batch.size(0), len(dose_times_1)))
                    elif batch[0, 6, 0] == 1:
                        dose_time_input = dose_times_2.expand((batch.size(0), len(dose_times_2)))
                    else:
                        dose_times_3 = np.random.choice(len(dose_times_2)) + 1
                        if dose_times_3 > 0:
                            dose_times_3 = dose_times_2[:dose_times_3]
                            dose_time_input = dose_times_3.expand((batch.size(0), len(dose_times_3)))
                        else:
                            dose_time_input = None
                    smooth_time_tensor = torch.linspace(full_time_tensor[0], full_time_tensor[-1], steps = int((full_time_tensor[-1] - full_time_tensor[0])))
                    x_hat = model.solve(covariate_input, dose_time_input, full_time_tensor).transpose(0, 1).squeeze()
                    if not batch[0, 6, 0] == 2:
                        total_loss += train_loss_func(x_hat, data_tensor) / (25 * 2 * data_tensor.size(-1))
            if total_loss < best_loss:
                if total_loss < 0.75:
                    batch_index = 0
                    for val_dataloader in val_data_loaders:
                        for batch in val_dataloader:
                            if batch_index == 1:
                                covariate_input = batch[:, 2:6, 0]
                                #covariate_input = batch[:, -1,] = (150 - np.mean(dose_sizes)) / np.std(dose_sizes)
                                covariate_input = covariate_input.to(torch.float32)
                                data_tensor = batch[:, 1, :].to(torch.float32)
                                dose_time_input = dose_times_2.expand((batch.size(0), len(dose_times_2)))
                                x_hat = model.solve(covariate_input, dose_time_input, smooth_time_tensor).transpose(0, 1).squeeze()
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
                                        smooth_time_tensor,
                                        x_hat.detach()[z, :],
                                        color='#1a6fcc',
                                        linewidth=2,
                                        alpha=0.95,
                                        label='Prediction $\\hat{x}$',
                                        zorder=3
                                    )

                                    # Subtle fill under the prediction curve
                                    """ ax.fill_between(
                                        smooth_time_tensor,
                                        x_hat.detach()[z, :],
                                        alpha=0.08,
                                        color='#1a6fcc',
                                        zorder=2
                                    ) """

                                    # Training data scatter
                                    ax.scatter(
                                        full_time_tensor,
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
                                    plt.savefig(f'Test_Data_Extrapolation{z}.pdf',  bbox_inches='tight', facecolor=fig.get_facecolor())
                        batch_index += 1
                #torch.save(model, 'flow_matching_model_swag_exponential_polynomial_11')
                best_loss = total_loss
            print(f'Validation Loss: {total_loss.item()}')
""" mean, cov = swag_tracker.finalize()
torch.save(mean, 'swag_mean_exponential_polynomial_11.pth')
torch.save(cov, 'swag_cov_exponential_polynomial_11.pth') """
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
                    
