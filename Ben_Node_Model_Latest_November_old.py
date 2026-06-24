import torchdiffeq as tdq
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Assuming each item in 'data' is a tuple or list of features
        return self.data[idx]

class Encoder(nn.Module):
    def __init__(self, hidden_layer, bottleneck):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(3, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, 4 * hidden_layer),
            nn.Linear(4 * hidden_layer,  bottleneck * 2)
        )
        self.kl= 0
        self.softplus = torch.nn.Softplus()
        self.bottleneck = bottleneck

    def forward(self, input):
        x = self.encoder(input)
        return x
    
class Decoder(nn.Module):
    def __init__(self, bottleneck):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck, 256, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(256, 64, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(64, 1, dtype=torch.float32)
        )
    def forward(self, latent_time_series):
        #input = torch.cat([latent_time_series[0], latent_time_series[1], latent_time_series[4].unsqueeze(1)], dim=1)
        latent_time_series_values = self.net(latent_time_series[0])
        return latent_time_series_values
    
class ODEFunc(nn.Module):
   def __init__(self, bottleneck):
       super(ODEFunc, self).__init__()
       self.net = nn.Sequential(
           nn.Linear(bottleneck * 3, 128), 
           nn.SiLU(),
           nn.Linear(128, 512),
           nn.SiLU(),
           nn.Linear(512, bottleneck)
       )
       self.dose_encoding = nn.Sequential(
           nn.Linear(3, 32), 
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )
   def forward(self, t, y):
        y0 = y[0]
        steering_variables = y[2]
        dosing_times = y[3]
        dosing_size = y[4].unsqueeze(0)
        t = t.unsqueeze(0)
        last_dose = t - torch.max(dosing_times[dosing_times <= t])
        total_doses = torch.tensor(len(dosing_times[dosing_times <= t]), dtype=torch.int).unsqueeze(0)
        dose_input = torch.cat([last_dose, total_doses, dosing_size])
        dose_encoding  = self.dose_encoding(dose_input) 
        ode_input = torch.cat([y0, steering_variables, dose_encoding])
        dzdt = self.net(ode_input)
        dzdt = torch.cat([dzdt, torch.zeros_like(dosing_times), torch.zeros_like(steering_variables), torch.zeros_like(y[1]), torch.zeros_like(dosing_size)])
        return dzdt

class NeuralODE(nn.Module):
   def __init__(self, ode_func):
       super(NeuralODE, self).__init__()
       self.ode_func = ode_func

   def forward(self, input, t):
       return odeint(self.ode_func, input, t, method='euler')

class LatentODE(nn.Module):
    def __init__(self, encoder, node, decoder):
       super(LatentODE, self).__init__()
       self.encoder = encoder
       self.node = node
       self.decoder = decoder
    def forward(self, input, predict_times, dosing_times):
        dose_size = input[-1]
        x = self.encoder(input[:-1])
        x = torch.split(x, len(x) // 2)
        y0 = torch.zeros_like(x[0])
        encoding = (y0, x[0], x[1], dosing_times, dose_size)
        latent_time_series = self.node(encoding, predict_times)
        decoded_time_series = self.decoder(latent_time_series)
        return decoded_time_series

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0)

def load_data(data_file, num_doses, covariate_file, dose_size):
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

    all_patient_data = np.zeros((75, 6, length))

    for i in range(1, 76):
        all_patient_data[i - 1, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 1, 1, :] = individual_patient_data[i]['CP']
        all_patient_data[i - 1, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 1, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 1, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 1, 5, :] = dose_size


    samples = np.zeros(29, dtype='int')
    weights = np.ones(29)
    weights[0] = 5

    i = 1
    weights[i] = 5
    weights[i + 1] = 20
    i += 2
    while i < num_doses  * 2 + 1:
        weights[i] = 2
        weights[i + 1] = 5
        i += 2
    
    i = 1
    samples[i] = samples[i - 1] + 20
    samples[i + 1] = samples[i] + 20
    samples[i + 2] = samples[i + 1] + 200
    i += 3
    while i < 26:
        samples[i] = samples[i - 1] + 40
        samples[i + 1] = samples[i] + 200
        i += 2
    samples[26] = samples[25] + 40
    samples[27] = samples[26] + 40
    samples[28] = samples[27] + 29

    all_patient_data = all_patient_data[:, :, samples]
    time = all_patient_data[0, 0, :]

    cp = all_patient_data[:, 1, :]

    time_mean = np.mean(time)
    time_std = np.std(time)
    cp_mean = np.mean(cp)
    cp_std = np.std(cp)

    time_mean = 0
    time_std = 1
    cp_mean = 0
    cp_std = 1

    normalized_time = (time - time_mean) / time_std 
    normalized_cp = (cp - cp_mean) /cp_std

    full_time_tensor = torch.tensor(normalized_time, dtype=torch.float32)
    check_time_tensor = torch.tensor(normalized_time, dtype=torch.float32)
    fine_time_tensor = torch.linspace(normalized_time[0], normalized_time[-1], 100)
    full_time_tensor = torch.cat([check_time_tensor, fine_time_tensor]).sort()[0].unique()
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return normalized_cp, full_time_tensor, mask, all_patient_data, weights

def load_validation_data(data_file, covariate_file, dose_size):
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

    all_patient_data = np.zeros((25, 6, length))

    for i in range(76, 101):
        all_patient_data[i - 76, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 76, 1, :] = individual_patient_data[i]['CP']
        all_patient_data[i - 76, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 76, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 76, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 76, 5, :] = dose_size

    samples = np.zeros(29, dtype='int')
    
    i = 1
    samples[i] = samples[i - 1] + 20
    samples[i + 1] = samples[i] + 20
    samples[i + 2] = samples[i + 1] + 200
    i += 3
    while i < 26:
        samples[i] = samples[i - 1] + 40
        samples[i + 1] = samples[i] + 200
        i += 2
    samples[26] = samples[25] + 40
    samples[27] = samples[26] + 40
    samples[28] = samples[27] + 29


    all_patient_data = all_patient_data[:, :, samples]
    time = all_patient_data[0, 0, :]

    cp = all_patient_data[:, 1, :]

    time_mean = np.mean(time)
    time_std = np.std(time)
    cp_mean = np.mean(cp)
    cp_std = np.std(cp)

    time_mean = 0
    time_std = 1
    cp_mean = 0
    cp_std = 1

    normalized_time = (time - time_mean) / time_std 
    normalized_cp = (cp - cp_mean) /cp_std

    full_time_tensor = torch.tensor(normalized_time, dtype=torch.float32)
    check_time_tensor = torch.tensor(normalized_time, dtype=torch.float32)
    fine_time_tensor = torch.linspace(normalized_time[0], normalized_time[-1], 100)
    full_time_tensor = torch.cat([check_time_tensor, fine_time_tensor]).sort()[0].unique()
    mask = torch.isin(full_time_tensor, check_time_tensor)
    return normalized_cp, full_time_tensor, mask, all_patient_data

def loss_func(x_hat, labels, weights):
    loss_weights = torch.tensor(weights, dtype=torch.float32)
    loss = torch.abs(x_hat - labels) * loss_weights
    loss = torch.mean(loss)
    return loss

dose_size_1 = 100
dose_size_2 = 50
dose_size_3 = 0

num_doses_1 = 5
num_doses_2 = 10

normalized_cp_1, full_time_tensor_1, mask_1, all_patient_data_1, weights = load_data('Data/PKPD_WT_AGE_SEX_100_24_5.csv', num_doses_1, 'Data/WT_AGE_SEX_100_24_5.csv', dose_size_1)
normalized_cp_2, full_time_tensor_2, mask_2, all_patient_data_2, weights = load_data('Data/PKPD_WT_AGE_SEX_50_24_10.csv', num_doses_2, 'Data/WT_AGE_SEX_50_24_10.csv', dose_size_2)
normalized_cp_3, full_time_tensor_3, mask_3, all_patient_data_3, weights = load_data('Data/PKPD_WT_AGE_SEX_0_24_10.csv', num_doses_2, 'Data/WT_AGE_SEX_0_24_10.csv', dose_size_3)


val_cp_1, val_time_tensor_1, val_mask_1, val_patient_data_1 = load_validation_data('Data/PKPD_WT_AGE_SEX_100_24_5.csv', 'Data/WT_AGE_SEX_100_24_5.csv', dose_size_1)
val_cp_2, val_time_tensor_2, val_mask_2, val_patient_data_2 = load_validation_data('Data/PKPD_WT_AGE_SEX_50_24_10.csv', 'Data/WT_AGE_SEX_50_24_10.csv', dose_size_2)
val_cp_3, val_time_tensor_3, val_mask_3, val_patient_data_3 = load_validation_data('Data/PKPD_WT_AGE_SEX_0_24_10.csv', 'Data/WT_AGE_SEX_0_24_10.csv', dose_size_3)

dose_times_1 = np.zeros(5)
dose_times_2 = np.zeros(10)

for i in range(1, 5):
    dose_times_1[i] = dose_times_1[i - 1] + 24
    dose_times_2[i] = dose_times_2[i - 1] + 24

for i in range(5, 10):
    dose_times_2[i] = dose_times_2[i - 1] + 24

dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)
dose_times_2 = torch.tensor(dose_times_2, dtype=torch.float32)


all_patient_data = np.vstack((all_patient_data_1, all_patient_data_2, all_patient_data_3))
patient_dataset = PatientDataset(all_patient_data)
batch_size = 32
dataloader = DataLoader(patient_dataset, batch_size=batch_size, shuffle=True)

val_patient_data = np.vstack((val_patient_data_1, val_patient_data_2, val_patient_data_3))
val_patient_dataset = PatientDataset(val_patient_data)
val_batch_size = 75
val_dataloader = DataLoader(val_patient_dataset, batch_size=val_batch_size, shuffle=False)

bottleneck = 16
encoder = Encoder(32, bottleneck)
decoder = Decoder(bottleneck)
ode_func = ODEFunc(bottleneck)
node = NeuralODE(ode_func)
latent_ODE = LatentODE(encoder, node, decoder)
latent_ODE.apply(initialize_weights)
#latent_ODE = torch.load('Models\WT_AGE_SEX_100_50_24_5_10_node_only')
epochs = 300
optimizer = torch.optim.Adam(latent_ODE.parameters(), lr=0.001)
best_loss = np.inf
cost_func = torch.nn.MSELoss()
for epoch in range(epochs):
    for batch in dataloader:
        total_loss = 0
        optimizer.zero_grad()
        for patient in batch:
            input = torch.cat([patient[2][0].unsqueeze(0), patient[3][0].unsqueeze(0), patient[4][0].unsqueeze(0), patient[5][0].unsqueeze(0)])
            input = input.to(torch.float32)
            data_tensor = patient[1]
            if patient[5][0] == dose_size_1:
                x_hat = latent_ODE(input, full_time_tensor_1, dose_times_1)
            elif patient[5][0] == dose_size_2:
                x_hat = latent_ODE(input, full_time_tensor_1, dose_times_2)
            else:
                dose_times_3 = np.random.choice(9)
                x_hat= latent_ODE(input, full_time_tensor_3, dose_times_2[:dose_times_3 + 1])
            x_hat = x_hat[mask_1]
            loss = loss_func(x_hat, data_tensor, weights)
            if torch.any(loss.isnan()):
                loss = torch.tensor(100, dtype=torch.float32)
                loss.requires_grad = True
            total_loss += loss / len(batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(latent_ODE.parameters(), max_norm=1.0)
        optimizer.step()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            total_loss = 0
            for patient in batch:
                input = torch.cat([patient[2][0].unsqueeze(0), patient[3][0].unsqueeze(0), patient[4][0].unsqueeze(0), patient[5][0].unsqueeze(0)])
                input = input.to(torch.float32)
                data_tensor = patient[1]
                if patient[5][0] == dose_size_1:
                    x_hat = latent_ODE(input, full_time_tensor_1, dose_times_1)
                elif patient[5][0] == dose_size_2:
                    x_hat = latent_ODE(input, full_time_tensor_1, dose_times_2)
                else:
                    dose_times_3 = np.random.choice(9)
                    x_hat= latent_ODE(input, full_time_tensor_3, dose_times_2[:dose_times_3 + 1])
                x_hat = x_hat[mask_1]
                loss = loss_func(x_hat, data_tensor, weights)
                if torch.any(loss.isnan()):
                    loss = torch.tensor(100, dtype=torch.float32)
                    loss.requires_grad = True
                total_loss += loss / val_batch_size
        print(total_loss.item())
    if total_loss.item() < best_loss:
        torch.save(latent_ODE, 'Models/WT_AGE_SEX_100_50_24_5_10_node_only_encoded_dose_info')
        best_loss = total_loss.item()
    if total_loss.item() < 0.1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001 

