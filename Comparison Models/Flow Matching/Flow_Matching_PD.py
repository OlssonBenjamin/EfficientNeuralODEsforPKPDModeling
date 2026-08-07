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

seed = 0
def load_parameters_into_model(model, param_vector):
    """
    Given a 1D param_vector,
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
    def __init__(self, data, cp_predictions):
        self.data = data
        self.cp_predictions = cp_predictions
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.cp_predictions[idx]

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
           nn.Linear(bottleneck + 3, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
       self.decreasing_net = nn.Sequential(
           nn.Linear(bottleneck + 2, 64),
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
            ode_input = torch.cat([steering_variables, last_dose, current_concentration], dim=1)
            dzdt = -self.decreasing_net(ode_input)
        else:
            ode_input = torch.cat([steering_variables, last_dose, current_concentration, dose_size], dim=1)
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
       self.std = 12.5
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
            self.std *= 0.999
    
    def solve(self, input, dosing_times, predict_times):
        x = self.patient_encoding(input[:, :-1])
        batch_size = x.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        final_time_series = nn.functional.relu(latent_time_series[:, :, 0])
        self.decay_std()
        return final_time_series
        

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0)




class PKPDSolver(nn.Module):
   def __init__(self, ode_func):
       super(PKPDSolver, self).__init__()
       self.ode_func = ode_func
       self.step_size = 0.1

   def forward(self, input, t):
       return odeint(self.ode_func, input, t, method='euler', options=dict(step_size=self.step_size))
   

class PKODE(nn.Module):
    def __init__(self, bottleneck, zero_dose_size):
        super(PKODE, self).__init__()
        self.increasing_net = nn.Sequential(
           nn.Linear(bottleneck + 3, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
        self.decreasing_net = nn.Sequential(
           nn.Linear(bottleneck + 2, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
        self.bottleneck = bottleneck
        self.zero_dose_size = zero_dose_size
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
            ode_input = torch.cat([steering_variables, last_dose, current_concentration], dim=1)
            dzdt = -self.decreasing_net(ode_input)
        else:
            ode_input = torch.cat([steering_variables, last_dose, current_concentration, dose_size], dim=1)
            dzdt = self.increasing_net(ode_input)
        if (dose_size == self.zero_dose_size).any():
            dzdt = torch.zeros_like(dzdt)
        dzdt = torch.cat([dzdt, torch.zeros_like(steering_variables), torch.zeros_like(dosing_times), torch.zeros_like(dose_size)], dim=-1)
        return dzdt

class PKPDODE(nn.Module):
   def __init__(self, bottleneck, zero_dose):
       super(PKPDODE, self).__init__()
       self.increasing_net = nn.Sequential(
           nn.Linear(bottleneck + 1, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
       self.decreasing_net = nn.Sequential(
           nn.Linear(bottleneck + 1, 64),
           nn.SiLU(),
           nn.Linear(64, 1),
           nn.Softplus()
       )
       self.concentration_encoding = nn.Sequential(
           nn.Linear(2, 64),
           nn.SiLU(),
           nn.Linear(64, bottleneck),
           nn.Softplus()
       )
       self.bottleneck = bottleneck
       self.peak_dose = 4.7
       self.baseline = 50
       self.pk_ode = PKODE(bottleneck, zero_dose)

   def forward(self, t, y):
        current_response = y[:, -1].unsqueeze(-1)
        current_concentration = y[:, 0].unsqueeze(-1)
        concentration_input = y[:, :-1]
        concentration_derivative = self.pk_ode(t, concentration_input)[:, 0].unsqueeze(-1)
        next_concentration = current_concentration + 0.1 * concentration_derivative
        negative_mask = next_concentration < 0
        next_concentration[negative_mask] = 0
        concentration_derivative[negative_mask] = 0
        dosing_times = y[:, self.bottleneck + 1:-2]
        t_expanded = t.unsqueeze(0).expand((current_concentration.size(0), dosing_times.size(1)))
        mask = (dosing_times <= t_expanded)
        masked_dosing_times = dosing_times * mask 
        last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]
        input = torch.cat([current_concentration, next_concentration], dim=-1)
        concentration_encoding = self.concentration_encoding(input)
        input = torch.cat([current_response, concentration_encoding], dim=-1)
        if (last_dose >= self.peak_dose).any():
            baseline = torch.zeros_like(current_response) + 50
            dzdt = self.increasing_net(input)  * (baseline - current_response)
            #* min((last_dose[0] - self.peak_dose) / 0.3, 1)**2
        else:
            dzdt = -self.decreasing_net(input) 
            #* min((self.peak_dose - last_dose[0]) / 1, 1)**2
        dzdt_final = torch.cat([concentration_derivative, torch.zeros_like(concentration_input[:, 1:]), dzdt], dim=-1)
        return dzdt_final

   
class PKPD(nn.Module):
    def __init__(self, num_variables, bottleneck, zero_dose):
       super(PKPD, self).__init__()
       self.patient_encoding = nn.Sequential(
           nn.Linear(num_variables, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )
       self.ode = PKPDODE(bottleneck, zero_dose)
       self.node = PKPDSolver(self.ode)
       self.std = 12.5
       self.zero_dose = zero_dose

    def forward(self, input, dosing_times, predict_times, trajectories, concentrations):
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
        
        interpolated_values = smaller_values + alpha * (bigger_values - smaller_values)

        initial_concentrations = concentrations[:, random_indices]

        if (input[0, -1] == self.zero_dose):
            interpolants = torch.normal(interpolated_values, 0.5)
        else:
            interpolants = torch.normal(interpolated_values, self.std)


        rt = random_times[0]

        interp_i = interpolants[:, 0].unsqueeze(1)

        encoding_i = torch.cat([
            initial_concentrations,
            x,
            dosing_times,
            input[:, -1].unsqueeze(-1),
            interp_i,
        ], dim=1)

        dzdt = self.ode(rt, encoding_i)[:, -1]

        t_expanded = rt.unsqueeze(0).expand((input.size(0), dosing_times.size(1)))
        mask = (dosing_times <= t_expanded)
        masked_dosing_times = dosing_times * mask 
        last_dose = rt - masked_dosing_times.max(dim=1, keepdim=True)[0]
        
        return dzdt, random_times, bigger_idx, smaller_idx, bigger_times, smaller_times, interpolants, last_dose
    

    def decay_std(self):
        if self.std >= 1:
            self.std *= 0.999
    
    def solve(self, input, dosing_times, predict_times):
        x = self.patient_encoding(input[:, :-1])
        batch_size = x.size(0)
        responses = torch.zeros(batch_size, 1, dtype=torch.float32) +  50
        concentrations = torch.zeros(batch_size, 1, dtype=torch.float32) 
        encoding = torch.cat([concentrations, x, dosing_times, input[:, -1].unsqueeze(-1), responses], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        final_time_series = latent_time_series[:, :, -1]
        #self.decay_std()
        return final_time_series
        

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0)

def load_data(data_file, covariate_file, dose_size, dose_regime_index, pk_model, dose_times):
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

    full_time_tensor = torch.tensor(individual_patient_data[1]['time'][1:]).to(torch.float32)

    for i in range(1, 76):
        all_patient_data[i - 1, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 1, 1, :] = individual_patient_data[i]['RESPONSE']
        all_patient_data[i - 1, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 1, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 1, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 1, 5, :] = dose_size
        all_patient_data[i - 1, 6, :] = dose_regime_index
        all_patient_data[i - 1, 7, :] = individual_patient_data[i]['CP']

    samples = np.zeros(35, dtype='int')
    
    i = 1
    samples[i] = samples[i - 1] + 48
    samples[i + 1] = samples[i] + 193
    i += 2
    while i < 35:
        samples[i] = samples[i - 1] + 47
        samples[i + 1] = samples[i] + 193
        i += 2
    samples[-1] = samples[-1] - 10
    sampled_patient_data = all_patient_data[:, :, samples]
    full_time_tensor = torch.tensor(all_patient_data[0, 0, :], dtype=torch.float32)
    check_time_tensor = torch.tensor(sampled_patient_data[0, 0, :], dtype=torch.float32)

    with torch.no_grad():
        if dose_size >= 0:
            covariates = torch.tensor(sampled_patient_data[:, 2:6, 0]).to(torch.float32)
            dosing_times = dose_times.expand(covariates.size(0), dose_times.size(0))
            cp_predictions = pk_model.solve(covariates, dosing_times, full_time_tensor).transpose(0, 1).squeeze()
        else:
            cp_predictions = torch.zeros(size=(75, 4081))
    return sampled_patient_data, cp_predictions

def load_validation_data(data_file, covariate_file, dose_size, dose_regime_index, pk_model, dose_times):
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

    all_patient_data = np.zeros((25, 8, length))

    full_time_tensor = torch.tensor(individual_patient_data[1]['time'][1:]).to(torch.float32)

    for i in range(76, 101):
        all_patient_data[i - 76, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 76, 1, :] = individual_patient_data[i]['RESPONSE']
        all_patient_data[i - 76, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 76, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 76, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 76, 5, :] = dose_size
        all_patient_data[i - 76, 6, :] = dose_regime_index
        all_patient_data[i - 76, 7, :] = individual_patient_data[i]['CP']

    samples = np.zeros(35, dtype='int')
    
    i = 1
    samples[i] = samples[i - 1] + 48
    samples[i + 1] = samples[i] + 193
    i += 2
    while i < 35:
        samples[i] = samples[i - 1] + 47
        samples[i + 1] = samples[i] + 193
        i += 2
    samples[-1] = samples[-1] - 10
    sampled_patient_data = all_patient_data[:, :, samples]

    with torch.no_grad():
        if dose_size >= 0:
            covariates = torch.tensor(sampled_patient_data[:, 2:6, 0]).to(torch.float32)
            dosing_times = dose_times.expand(covariates.size(0), dose_times.size(0))
            cp_predictions = pk_model.solve(covariates, dosing_times, full_time_tensor).transpose(0, 1).squeeze()
        else:
            cp_predictions = torch.zeros(size=(75, 4081))

    return sampled_patient_data, cp_predictions


dose_size_1 = 100
dose_size_2 = 50
dose_size_3 = 0

dose_sizes = np.array([100, 50, 0])

dose_size_1  = (dose_size_1 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_2  = (dose_size_2 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_3  = (dose_size_3 - np.mean(dose_sizes)) / np.std(dose_sizes)

dose_times_1 = np.zeros(5)
dose_times_2 = np.zeros(10)

for i in range(1, 5):
    dose_times_1[i] = dose_times_1[i - 1] + 24
    dose_times_2[i] = dose_times_2[i - 1] + 24

for i in range(5, 10):
    dose_times_2[i] = dose_times_2[i - 1] + 24

dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)
dose_times_2 = torch.tensor(dose_times_2, dtype=torch.float32)

bottleneck = 4
pk_model = TrainODE(3, bottleneck, dose_size_3)
pk_model.load_state_dict(torch.load(f'Saved Weights/flow_matching_model_{seed}', weights_only=True))


all_patient_data_1, cp_predictions_1 = load_data('Training Data/PK_PD_100_24_5_100_WT_AGE_SEX.csv', 'Training Data/100_24_5_100_WT_AGE_SEX.csv', dose_size_1, 0, pk_model, dose_times_1)
all_patient_data_2, cp_predictions_2  = load_data('Training Data/PK_PD_50_24_10_100_WT_AGE_SEX.csv', 'Training Data/50_24_10_100_WT_AGE_SEX.csv', dose_size_2, 1, pk_model, dose_times_2)
all_patient_data_3, cp_predictions_3 = load_data('Training Data/PK_PD_0_24_10_100_WT_AGE_SEX.csv', 'Training Data/0_24_10_100_WT_AGE_SEX.csv', dose_size_3, 2, pk_model, dose_times_1)

val_patient_data_1, val_cp_predictions_1 = load_validation_data('Training Data/PK_PD_100_24_5_100_WT_AGE_SEX.csv', 'Training Data/100_24_5_100_WT_AGE_SEX.csv', dose_size_1, 0, pk_model, dose_times_1)
val_patient_data_2, val_cp_predictions_2 = load_validation_data('Training Data/PK_PD_50_24_10_100_WT_AGE_SEX.csv', 'Training Data/50_24_10_100_WT_AGE_SEX.csv', dose_size_2, 1, pk_model, dose_times_2)
val_patient_data_3, val_cp_predictions_3 = load_validation_data('Training Data/PK_PD_0_24_10_100_WT_AGE_SEX.csv', 'Training Data/0_24_10_100_WT_AGE_SEX.csv', dose_size_3, 2, pk_model, dose_times_1)

batch_size = 8
patient_dataset_1 = PatientDataset(all_patient_data_1, cp_predictions_1)
patient_dataset_2 = PatientDataset(all_patient_data_2, cp_predictions_2)
patient_dataset_3 = PatientDataset(all_patient_data_3, cp_predictions_3)
dataloader_1 = DataLoader(patient_dataset_1, batch_size=batch_size, shuffle=True)
dataloader_2 = DataLoader(patient_dataset_2, batch_size=batch_size, shuffle=True)
dataloader_3 = DataLoader(patient_dataset_3, batch_size=batch_size, shuffle=True)

val_batch_size = 25
val_patient_dataset_1 = PatientDataset(val_patient_data_1, val_cp_predictions_1)
val_patient_dataset_2 = PatientDataset(val_patient_data_2, val_cp_predictions_2)
val_patient_dataset_3 = PatientDataset(val_patient_data_3, val_cp_predictions_3)
val_dataloader_1 = DataLoader(val_patient_dataset_1, batch_size=val_batch_size, shuffle=False)
val_dataloader_2 = DataLoader(val_patient_dataset_2, batch_size=val_batch_size, shuffle=False)
val_dataloader_3 = DataLoader(val_patient_dataset_3, batch_size=val_batch_size, shuffle=False)


train_dataloaders = [dataloader_1, dataloader_2, dataloader_3]
val_data_loaders = [val_dataloader_1, val_dataloader_2, val_dataloader_3]

model = PKPD(3, bottleneck, dose_size_3)
model.apply(initialize_weights)
epochs = 4001
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
best_loss = np.inf
cost_func = torch.nn.MSELoss()
train_loss_func = torch.nn.MSELoss(reduction='sum')
training_loss = np.zeros(epochs)
validation_loss = np.zeros(epochs)
length_training_data = len(all_patient_data_1) + len(all_patient_data_2) + len(all_patient_data_3)



model.patient_encoding = pk_model.patient_encoding

model.ode.pk_ode.increasing_net = pk_model.ode.increasing_net

model.ode.pk_ode.decreasing_net = pk_model.ode.decreasing_net

for parameter in model.patient_encoding.parameters():
    parameter.requires_grad = False

for parameter in model.ode.pk_ode.parameters():
    parameter.requires_grad = False

for epoch in range(epochs):
    train_loss = 0
    for train_dataloader in train_dataloaders:
        for input_batch, cp_batch in train_dataloader:
            total_loss = 0
            optimizer.zero_grad()
            covariate_input = input_batch[:, 2:6, 0].to(torch.float32)
            data_tensor = input_batch[:, 1, :].to(torch.float32)
            full_time_tensor = input_batch[0, 0, :].to(torch.float32)
            if input_batch[0, 6, 0] == 0:
                dose_time_input = dose_times_1.expand((input_batch.size(0), len(dose_times_1)))
            elif input_batch[0, 6, 0] == 1:
                dose_time_input = dose_times_2.expand((input_batch.size(0), len(dose_times_2)))
            else:
                dose_times_3 = np.random.choice(len(dose_times_2)) + 1
                if dose_times_3 > 0:
                    dose_times_3 = dose_times_2[:dose_times_3]
                    dose_time_input = dose_times_3.expand((input_batch.size(0), len(dose_times_3)))
                else:
                    dose_time_input = None
            dzdt, random_times, bigger_idx, smaller_idx, bigger_times, smaller_times, interpolants, last_dose = model(covariate_input, dose_time_input, full_time_tensor, data_tensor, cp_batch)
            if last_dose[0] >= 4.7 and not input_batch[0, 6, 0] == 2:
                baseline = torch.zeros_like(interpolants) + 50  
                delta_t = bigger_times - smaller_times          
                
                numerator = baseline - data_tensor[:, bigger_idx]     
                denominator = baseline - data_tensor[:, smaller_idx]    
                
                numerator_mask = (numerator == 0)
                denominator_mask = (denominator == 0)

                numerator = numerator + numerator_mask.float() * 1e-8
                denominator = denominator + denominator_mask.float() * 1e-8

                k = -torch.log(numerator / denominator) / delta_t   
                
                conditional_field = k * (baseline - interpolants) 
                #* min((last_dose[0] - 4.5) / 1, 1)**2
            else:
                if not input_batch[0, 6, 0] == 2:
                    multiplier = min((4.7 - last_dose[0]) / 1, 1)**2
                else:
                    multiplier = 1
                conditional_field = torch.tensor((data_tensor[:, bigger_idx] - interpolants) / (bigger_times - random_times)).to(torch.float32)
            #if input_batch[0, 6, 0] == 2:
                #conditional_field = interpolants * 0 
            conditional_field = conditional_field.squeeze()
            loss = cost_func(dzdt, conditional_field)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            train_loss += loss
    if epoch == 2000:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    training_loss[epoch] = train_loss / (length_training_data * len(full_time_tensor))
    with torch.no_grad():
        total_loss = 0
        if (epoch) % 100 == 0:
            print(f'Epoch: {epoch}')
            print(f'Training Loss: {train_loss.item()}')
            for val_dataloader in val_data_loaders:
                for input_batch, cp_batch in val_dataloader:
                    covariate_input = input_batch[:, 2:6, 0].to(torch.float32)
                    data_tensor = input_batch[:, 1, :].to(torch.float32)
                    full_time_tensor = input_batch[0, 0, :].to(torch.float32)
                    concentrations = input_batch[:, 7, :].to(torch.float32)
                    if input_batch[0, 6, 0] == 0:
                        dose_time_input = dose_times_1.expand((input_batch.size(0), len(dose_times_1)))
                    elif input_batch[0, 6, 0] == 1:
                        dose_time_input = dose_times_2.expand((input_batch.size(0), len(dose_times_2)))
                    else:
                        dose_times_3 = np.random.choice(len(dose_times_2)) + 1
                        if dose_times_3 > 0:
                            dose_times_3 = dose_times_2[:dose_times_3]
                            dose_time_input = dose_times_3.expand((input_batch.size(0), len(dose_times_3)))
                        else:
                            dose_time_input = None
                    x_hat = model.solve(covariate_input, dose_time_input, full_time_tensor).transpose(0, 1).squeeze()
                    total_loss += train_loss_func(x_hat, data_tensor) / (25 * 2 * data_tensor.size(-1))
            print(f'Validation Loss: {total_loss.item()}')
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save(model.state_dict(), f'Saved Weights/flow_matching_pkpd_concentration_{seed}')
