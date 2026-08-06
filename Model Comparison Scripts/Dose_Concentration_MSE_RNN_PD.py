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
        return self.data[idx]

class RNN(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
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
    
    def forward(self, initial_conc, steering_variables, dosing_times, time_steps, dose_size):
        output_trajectory = [initial_conc.squeeze()]
        conc_t = initial_conc.clone()
        for t in time_steps[1:]:
            t_expanded = t.unsqueeze(0).expand(dosing_times.shape)
            mask = (dosing_times <= t_expanded)
            masked_dosing_times = dosing_times * mask
            last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]
            dose_input = torch.cat([last_dose, dose_size], dim=1)
            dose_encoding = self.dose_encoding(dose_input)
            net_input = torch.cat([conc_t, steering_variables, dose_encoding], dim=1)
            dconc = self.net(net_input)
            room_up = (50.0 - conc_t) / 50.0
            room_down = conc_t / 50.0
            sign_mask = dconc > 0
            time_mask = last_dose > 24.0
            dconc = torch.where(sign_mask & time_mask, dconc * room_up, dconc)
            dconc = torch.where(~sign_mask, dconc * room_down, dconc)
            conc_t = conc_t + dconc
            output_trajectory.append(conc_t.squeeze())
        return torch.stack(output_trajectory, dim=0)
    
class PKPDPredictor(nn.Module):
    def __init__(self, num_variables, bottleneck):
       super(PKPDPredictor, self).__init__()
       self.rnn = RNN(bottleneck)
       self.patient_encoding = nn.Sequential(
           nn.Linear(num_variables, 32),
           nn.SiLU(),
           nn.Linear(32, bottleneck)
       )

    def forward(self, input, dosing_times, predict_times):
        batch_size = input.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32) + 50
        steering_variables = self.patient_encoding(input[:, :-1])
        time_series = self.rnn(y0, steering_variables, dosing_times, predict_times, input[:, -1].unsqueeze(-1))
        return torch.abs(time_series)
    
def prepare_initial_input(batch_size, sequence_length):
    return torch.zeros(batch_size, sequence_length, 1)  # Start with zeros or any initial value

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


seed = 4
model = torch.load(f'Saved Models/rnn_pd_{seed}', weights_only=False)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

full_time_tensor, val_mask, val_patient_data_1 = load_data('Dose Size Data/0_24_8_100.csv', 'Dose Size Data/PK_PD_0_24_8_100.csv', dose_size_1, 0)
full_time_tensor, val_mask, val_patient_data_2 = load_data('Dose Size Data/25_24_8_100.csv', 'Dose Size Data/PK_PD_25_24_8_100.csv', dose_size_2, 0)
full_time_tensor, val_mask, val_patient_data_3 = load_data('Dose Size Data/50_24_8_100.csv', 'Dose Size Data/PK_PD_50_24_8_100.csv', dose_size_3, 0)
full_time_tensor, val_mask, val_patient_data_4 = load_data('Dose Size Data/75_24_8_100.csv', 'Dose Size Data/PK_PD_75_24_8_100.csv', dose_size_4, 0)
full_time_tensor, val_mask, val_patient_data_5 = load_data('Dose Size Data/100_24_8_100.csv', 'Dose Size Data/PK_PD_100_24_8_100.csv', dose_size_5, 0)
full_time_tensor, val_mask, val_patient_data_6 = load_data('Dose Size Data/125_24_8_100.csv', 'Dose Size Data/PK_PD_125_24_8_100.csv', dose_size_6, 0)
full_time_tensor, val_mask, val_patient_data_7 = load_data('Dose Size Data/150_24_8_100.csv', 'Dose Size Data/PK_PD_150_24_8_100.csv', dose_size_7, 0)
full_time_tensor, val_mask, val_patient_data_8 = load_data('Dose Size Data/175_24_8_100.csv', 'Dose Size Data/PK_PD_175_24_8_100.csv', dose_size_8, 0)
full_time_tensor, val_mask, val_patient_data_9 = load_data('Dose Size Data/200_24_8_100.csv', 'Dose Size Data/PK_PD_200_24_8_100.csv', dose_size_9, 0)
full_time_tensor, val_mask, val_patient_data_10 = load_data('Dose Size Data/225_24_8_100.csv', 'Dose Size Data/PK_PD_225_24_8_100.csv', dose_size_10, 0)
full_time_tensor, val_mask, val_patient_data_11 = load_data('Dose Size Data/250_24_8_100.csv', 'Dose Size Data/PK_PD_250_24_8_100.csv', dose_size_11, 0)


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

dose_times_1 = 4 * np.zeros(9)


for i in range(1, 9):
    dose_times_1[i] = dose_times_1[i - 1] + 24


dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)

mse_list = []
model.eval()
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
np.save(f'rnn_pd_dose_size_{seed}',np.array(mse_list))
