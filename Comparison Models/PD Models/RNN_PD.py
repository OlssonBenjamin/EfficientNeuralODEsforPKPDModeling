import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

#Seeds used to generate models were 0, 1, 3, 4, 5
seed = 5
torch.manual_seed(seed)
np.random.seed(seed)

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
        return time_series
    
def prepare_initial_input(batch_size, sequence_length):
    return torch.zeros(batch_size, sequence_length, 1)  # Start with zeros or any initial value

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
        all_patient_data[i - 76, 1, :] = individual_patient_data[i]['RESPONSE']
        all_patient_data[i - 76, 2, :] = individual_patient_covariate_data[i]['WT']
        all_patient_data[i - 76, 3, :] = individual_patient_covariate_data[i]['AGE']
        all_patient_data[i - 76, 4, :] = individual_patient_covariate_data[i]['SEX']
        all_patient_data[i - 76, 5, :] = dose_size
        all_patient_data[i - 76, 6, :] = dose_regime_index

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

full_time_tensor, mask, all_patient_data_1 = load_data('Random Data/PK_PD_100_24_10_100_1.csv', 'Random Data/100_24_10_100_1.csv', dose_size_1, 0)
full_time_tensor, mask, all_patient_data_2 = load_data('Random Data/PK_PD_50_24_10_100_1.csv', 'Random Data/50_24_10_100_1.csv', dose_size_2, 1)
full_time_tensor, mask, all_patient_data_3 = load_data('Random Data/PK_PD_0_24_10_100_1.csv', 'Random Data/0_24_10_100_1.csv', dose_size_3, 2)


val_time_tensor_1, val_mask_1, val_patient_data_1 = load_validation_data('Random Data/PK_PD_100_24_10_100_1.csv', 'Random Data/100_24_10_100_1.csv', dose_size_1, 0)
val_time_tensor_2, val_mask_2, val_patient_data_2 = load_validation_data('Random Data/PK_PD_50_24_10_100_1.csv', 'Random Data/50_24_10_100_1.csv', dose_size_2, 1)
val_time_tensor_3, val_mask_3, val_patient_data_3 = load_validation_data('Random Data/PK_PD_0_24_10_100_1.csv', 'Random Data/0_24_10_100_1.csv', dose_size_3, 2)


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
model = PKPDPredictor(3, bottleneck)
model.apply(initialize_weights)
epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
best_loss = np.inf
cost_func = torch.nn.MSELoss()
train_loss_func = torch.nn.MSELoss(reduction='sum')
training_loss = np.zeros(epochs)
validation_loss = np.zeros(epochs)
length_training_data = len(all_patient_data_1) + len(all_patient_data_2) + len(all_patient_data_3)

for epoch in range(epochs):
    train_loss = 0
    for train_dataloader in train_dataloaders:
        for batch in train_dataloader:
            total_loss = 0
            optimizer.zero_grad()
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
            x_hat = model(covariate_input, dose_time_input, full_time_tensor)[mask].transpose(0, 1)
            train_loss += train_loss_func(x_hat, data_tensor) 
            loss = cost_func(x_hat, data_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    training_loss[epoch] = (train_loss / (length_training_data * len(full_time_tensor))).detach().item()
    with torch.no_grad():
        total_loss = 0
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
                x_hat = model(covariate_input, dose_time_input, full_time_tensor)[mask].transpose(0, 1)
                total_loss += train_loss_func(x_hat, data_tensor) / (val_batch_size * len(train_dataloaders) * len(full_time_tensor))
        print(total_loss.item())
        validation_loss[epoch] = total_loss.item()
    if total_loss.item() < best_loss:
        torch.save(model, f'Saved Models/rnn_pd_{seed}')
        best_loss = total_loss.item()
        print("New Best")
    scheduler.step()
print(f"Best Loss: {best_loss} using seed: {seed}")
