# Imports
from torchdiffeq import odeint
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# Dataset wrapper
class PatientDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ODE solver wrapper
class NeuralODE(nn.Module):
   def __init__(self, ode_func):
       super(NeuralODE, self).__init__()
       self.ode_func = ode_func
       self.step_size = 0.1

   def forward(self, input, t):
       return odeint(
           self.ode_func,
           input,
           t,
           method='euler',
           options=dict(step_size=self.step_size)
       )


# ODE dynamics
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

        # Time since latest dose
        t_expanded = t.unsqueeze(0).expand(
            (current_concentration.size(0), dosing_times.size(1))
        )

        mask = dosing_times <= t_expanded
        masked_dosing_times = dosing_times * mask
        last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]

        # Choose uptake or decay
        if (last_dose >= self.peak_dose).any():
            ode_input = torch.cat(
                [steering_variables, last_dose, current_concentration],
                dim=1
            )
            dzdt = -self.decreasing_net(ode_input)
        else:
            ode_input = torch.cat(
                [steering_variables, last_dose, current_concentration, dose_size],
                dim=1
            )
            dzdt = self.increasing_net(ode_input)

        dzdt = torch.cat(
            [
                dzdt,
                torch.zeros_like(steering_variables),
                torch.zeros_like(dosing_times),
                torch.zeros_like(dose_size)
            ],
            dim=-1
        )

        return dzdt


# Full model
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

    def forward(self, input, dosing_times, predict_times, trajectories):
        x = self.patient_encoding(input[:, :-1])

        # Fine time grid
        fine_predict_times = torch.linspace(
            predict_times[0],
            predict_times[-1],
            steps=int(predict_times[-1] * 10 + 1)
        )

        mask = ~torch.isin(fine_predict_times, predict_times[-1])
        fine_predict_times = fine_predict_times[mask]

        num_samples = 1

        # Sample interpolation times
        random_indices = torch.randint(
            low=0,
            high=len(fine_predict_times),
            size=(num_samples,)
        )

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

        # Interpolate trajectory values
        alpha = (random_times - smaller_times) / (bigger_times - smaller_times)
        alpha = alpha.unsqueeze(0).expand(batch_size, -1)

        interpolated_values = (
            smaller_values + alpha * (bigger_values - smaller_values)
        )

        if input[0, -1] == self.zero_dose:
            interpolants = torch.normal(interpolated_values, 0.5)
        else:
            interpolants = torch.normal(interpolated_values, self.std)

        dzdt_list = []

        # Evaluate dynamics at sampled times
        for idx in range(num_samples):
            rt = random_times[idx]
            interp_i = interpolants[:, idx].unsqueeze(1)

            encoding_i = torch.cat(
                [
                    interp_i,
                    x,
                    dosing_times,
                    input[:, -1].unsqueeze(-1)
                ],
                dim=1
            )

            dzdt_i = self.ode(rt, encoding_i)[:, 0]
            dzdt_list.append(dzdt_i)

        dzdt_all = torch.stack(dzdt_list, dim=1)

        return dzdt_all, random_times, bigger_idx, bigger_times, interpolants

    def decay_std(self):
        if self.std >= 1:
            self.std *= 0.99
    
    def solve(self, input, dosing_times, predict_times):
        x = self.patient_encoding(input[:, :-1])

        batch_size = x.size(0)
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32)

        # Initial ODE state
        encoding = torch.cat(
            [
                y0,
                x,
                dosing_times,
                input[:, -1].unsqueeze(-1)
            ],
            dim=1
        )

        latent_time_series = self.node(encoding, predict_times)
        final_time_series = nn.functional.relu(latent_time_series[:, :, 0])

        self.decay_std()

        return final_time_series


# Weight initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0)


# Load training data
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

    # Select time samples
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


# Load validation data
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

    # Select time samples
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


# Dose normalization
dose_size_1 = 100
dose_size_2 = 50
dose_size_3 = 0

dose_sizes = np.array([100, 50, 0])

dose_size_1 = (dose_size_1 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_2 = (dose_size_2 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_3 = (dose_size_3 - np.mean(dose_sizes)) / np.std(dose_sizes)


full_time_tensor, mask, all_patient_data_1 = load_data('Training Data/PK_PD_100_24_5_100_WT_AGE_SEX.csv', 'Training Data/100_24_5_100_WT_AGE_SEX.csv', dose_size_1, 0)
full_time_tensor, mask, all_patient_data_2 = load_data('Training Data/PK_PD_50_24_10_100_WT_AGE_SEX.csv', 'Training Data/50_24_10_100_WT_AGE_SEX.csv', dose_size_2, 1)
full_time_tensor, mask, all_patient_data_3 = load_data('Training Data/PK_PD_0_24_10_100_WT_AGE_SEX.csv', 'Training Data/0_24_10_100_WT_AGE_SEX.csv', dose_size_3, 2)


val_time_tensor_1, val_mask_1, val_patient_data_1 = load_validation_data('Training Data/PK_PD_100_24_5_100_WT_AGE_SEX.csv', 'Training Data/100_24_5_100_WT_AGE_SEX.csv', dose_size_1, 0)
val_time_tensor_2, val_mask_2, val_patient_data_2 = load_validation_data('Training Data/PK_PD_50_24_10_100_WT_AGE_SEX.csv', 'Training Data/50_24_10_100_WT_AGE_SEX.csv', dose_size_2, 1)
val_time_tensor_3, val_mask_3, val_patient_data_3 = load_validation_data('Training Data/PK_PD_0_24_10_100_WT_AGE_SEX.csv', 'Training Data/0_24_10_100_WT_AGE_SEX.csv', dose_size_3, 2)


# Dose schedules
dose_times_1 = np.zeros(5)
dose_times_2 = np.zeros(10)

for i in range(1, 5):
    dose_times_1[i] = dose_times_1[i - 1] + 24
    dose_times_2[i] = dose_times_2[i - 1] + 24

for i in range(5, 10):
    dose_times_2[i] = dose_times_2[i - 1] + 24

dose_times_1 = torch.tensor(dose_times_1, dtype=torch.float32)
dose_times_2 = torch.tensor(dose_times_2, dtype=torch.float32)


# DataLoaders
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

val_dataloader_1 = DataLoader(
    val_patient_dataset_1,
    batch_size=val_batch_size,
    shuffle=False
)

val_dataloader_2 = DataLoader(
    val_patient_dataset_2,
    batch_size=val_batch_size,
    shuffle=False
)

val_dataloader_3 = DataLoader(
    val_patient_dataset_3,
    batch_size=val_batch_size,
    shuffle=False
)


train_dataloaders = [dataloader_1, dataloader_2, dataloader_3]
val_data_loaders = [val_dataloader_1, val_dataloader_2, val_dataloader_3]


# Model setup
bottleneck = 4

model = TrainODE(3, bottleneck, dose_size_3)
model.apply(initialize_weights)

epochs = 2501

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-3
)

best_loss = np.inf

cost_func = torch.nn.MSELoss()
train_loss_func = torch.nn.MSELoss(reduction='sum')

training_loss = np.zeros(epochs)
validation_loss = np.zeros(11)

length_training_data = (
    len(all_patient_data_1)
    + len(all_patient_data_2)
    + len(all_patient_data_3)
)


# Training loop
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
                dose_time_input = dose_times_1.expand(
                    (batch.size(0), len(dose_times_1))
                )
            elif batch[0, 6, 0] == 1:
                dose_time_input = dose_times_2.expand(
                    (batch.size(0), len(dose_times_2))
                )
            else:
                dose_times_3 = np.random.choice(len(dose_times_2)) + 1

                if dose_times_3 > 0:
                    dose_times_3 = dose_times_2[:dose_times_3]
                    dose_time_input = dose_times_3.expand(
                        (batch.size(0), len(dose_times_3))
                    )
                else:
                    dose_time_input = None

            dzdt, random_times, bigger_idx, bigger_times, interpolants = model(
                covariate_input,
                dose_time_input,
                full_time_tensor,
                data_tensor
            )

            # Finite-difference target
            conditional_field = torch.tensor(
                (data_tensor[:, bigger_idx] - interpolants)
                / (bigger_times - random_times)
            ).to(torch.float32)

            loss = cost_func(dzdt, conditional_field)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            train_loss += loss
    training_loss[epoch] = train_loss.item()
    # Validation
    with torch.no_grad():
        total_loss = 0

        if epoch % 250 == 0:
            print(f'Epoch: {epoch}')
            print(f'Training Loss: {train_loss.item()}')

            for val_dataloader in val_data_loaders:
                for batch in val_dataloader:
                    covariate_input = batch[:, 2:6, 0]
                    covariate_input = covariate_input.to(torch.float32)

                    data_tensor = batch[:, 1, :].to(torch.float32)

                    if batch[0, 6, 0] == 0:
                        dose_time_input = dose_times_1.expand(
                            (batch.size(0), len(dose_times_1))
                        )
                    elif batch[0, 6, 0] == 1:
                        dose_time_input = dose_times_2.expand(
                            (batch.size(0), len(dose_times_2))
                        )
                    else:
                        dose_times_3 = np.random.choice(len(dose_times_2)) + 1

                        if dose_times_3 > 0:
                            dose_times_3 = dose_times_2[:dose_times_3]
                            dose_time_input = dose_times_3.expand(
                                (batch.size(0), len(dose_times_3))
                            )
                        else:
                            dose_time_input = None

                    x_hat = model.solve(
                        covariate_input,
                        dose_time_input,
                        full_time_tensor
                    ).transpose(0, 1).squeeze()

                    total_loss += train_loss_func(x_hat, data_tensor) / (
                        25 * 3 * data_tensor.size(-1)
                    )

            print(f'Validation Loss: {total_loss.item()}')
            validation_loss[epoch // 250] = total_loss.item()
            if total_loss < best_loss:
                torch.save(model.state_dict(), 'Saved Weights/flow_matching_model_1')
                best_loss = total_loss
