from torchdiffeq import odeint
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


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
        dzdt = torch.cat([dzdt, torch.zeros_like(steering_variables), torch.zeros_like(dosing_times), torch.zeros_like(dose_size)], dim=1)
        return dzdt

class NeuralODE(nn.Module):
   def __init__(self, ode_func):
       super(NeuralODE, self).__init__()
       self.ode_func = ode_func

   def forward(self, input, t):
       return odeint(self.ode_func, input, t, method='euler',
                     options=dict(step_size=1.0)
                     )

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
        y0 = torch.zeros(batch_size, 1, dtype=torch.float32)
        encoding = torch.cat([y0, x, dosing_times, input[:, -1].unsqueeze(-1)], dim=1)
        latent_time_series = self.node(encoding, predict_times)
        return latent_time_series[:, :, 0]

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

    all_patient_data = np.zeros((100, 7, length))

    for i in range(1, 101):
        all_patient_data[i - 1, 0, :] = individual_patient_data[i]['time']
        all_patient_data[i - 1, 1, :] = individual_patient_data[i]['CP']
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

class TestNetwork(nn.Module):
    def __init__(self, patient_encoding, dose_encoding, net):
       super(TestNetwork, self).__init__()
       self.patient_encoding = patient_encoding
       self.dose_encoding = dose_encoding
       self.net = net

    def forward(self, input):
        covariates = input[:, :3]
        dose_size = input[:, 3].unsqueeze(-1)
        last_dose = input[:, 4].unsqueeze(-1)
        current_concentration = input[:, 5].unsqueeze(-1)
        steering_variables = self.patient_encoding(covariates)
        dose_input = torch.cat([last_dose, dose_size], dim=1)
        dose_encoding = self.dose_encoding(dose_input)
        ode_input = torch.cat([current_concentration, steering_variables, dose_encoding], dim=1)
        dzdt = self.net(ode_input)
        return dzdt

dose_size_1 = 100
dose_size_2 = 50
dose_size_3 = 0

dose_sizes = np.array([100, 50, 0])

dose_size_1  = (dose_size_1 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_2  = (dose_size_2 - np.mean(dose_sizes)) / np.std(dose_sizes)
dose_size_3  = (dose_size_3 - np.mean(dose_sizes)) / np.std(dose_sizes)

full_time_tensor, mask, all_patient_data_1 = load_data('Non Random Data/PKPD_100_24_5_Non_Random.csv', 'Non Random Data/100_24_5_Non_Random.csv', dose_size_1, 0)
full_time_tensor, mask, all_patient_data_2 = load_data('Non Random Data/PKPD_50_24_10__Non_Random.csv', 'Non Random Data/50_24_10_Non_Random.csv', dose_size_2, 1)
full_time_tensor, mask, all_patient_data_3 = load_data('Non Random Data/PKPD_0_24_10_Non_Random.csv', 'Non Random Data/0_24_10_Non_Random.csv', dose_size_3, 2)

last_dose = torch.tensor(0.0, dtype=torch.float32).expand([225, 1])
current_concentration = torch.tensor(0.0, dtype=torch.float32).expand([225, 1])

input_1 = all_patient_data_1[:, 2:6, 0]
input_2 = all_patient_data_2[:, 2:6, 0]
input_3 = all_patient_data_3[:, 2:6, 0]
input = np.vstack([input_1, input_2, input_3])

input = torch.tensor(input)
input_tensor = torch.cat([input, last_dose, current_concentration], dim=1).to(torch.float32)
input = input_tensor.numpy()

column_names = ['Weight', 'Age', 'Sex', 'Dose Size', 'Last Dose', 'Current Concentration']

df = pd.DataFrame(input, columns=column_names)
seed = 0
bottleneck = 16
node = NeuralODE(ODEFunc(bottleneck))
model = LatentODE(3, node, bottleneck)
model.load_state_dict(torch.load(f'Saved Weights/node_backprop_solver_pk_{seed}', weights_only=True))

patient_encoding_network = model.patient_encoding
dose_encoding_network = model.node.ode_func.dose_encoding
net = model.node.ode_func.net
shap_model = TestNetwork(patient_encoding_network, dose_encoding_network, net)

explainer = shap.GradientExplainer(shap_model, input_tensor)

# Average covariate SHAP values across several time-since-last-dose points
# (one full dosing interval, hourly), and drop Last Dose / Current Concentration
# from the summary plot since they're being marginalized over rather than shown.
covariates_tensor = input_tensor[:, :4]
time_since_last_dose = torch.linspace(0.0, 24.0, steps=25)

# Reference dosing schedules matching the loaded groups (100mg/5-dose, 50mg & 0mg/10-dose)
dose_times_1 = torch.zeros(5, dtype=torch.float32)
dose_times_2 = torch.zeros(10, dtype=torch.float32)
for i in range(1, 5):
    dose_times_1[i] = dose_times_1[i - 1] + 24
    dose_times_2[i] = dose_times_2[i - 1] + 24
for i in range(5, 10):
    dose_times_2[i] = dose_times_2[i - 1] + 24

group_sizes = [75, 75, 75]
group_dose_times = [dose_times_1, dose_times_2, dose_times_2]

# Solve each group's trajectory (model.forward drives the ODE via odeint) from the
# last scheduled dose through the following 24h, so column j is the concentration
# at time_since_last_dose[j].
concentration_trajectories = []
start = 0
with torch.no_grad():
    for size, dt in zip(group_sizes, group_dose_times):
        group_covariates = covariates_tensor[start:start + size]
        dosing_times_batch = dt.unsqueeze(0).expand(size, -1)
        predict_times = dt[-1] + time_since_last_dose
        trajectory = model(group_covariates, dosing_times_batch, predict_times).transpose(0, 1)
        concentration_trajectories.append(trajectory)
        start += size
concentration_trajectory = torch.cat(concentration_trajectories, dim=0)

covariate_shap_values = []
for j, t in enumerate(time_since_last_dose):
    last_dose_t = torch.full((225, 1), t.item(), dtype=torch.float32)
    current_concentration_t = concentration_trajectory[:, j].unsqueeze(-1)
    input_tensor_t = torch.cat([covariates_tensor, last_dose_t, current_concentration_t], dim=1)
    shap_values_t = explainer.shap_values(input_tensor_t)[:, :, 0]
    covariate_shap_values.append(shap_values_t[:, :3])

covariate_shap_values = np.mean(np.stack(covariate_shap_values, axis=0), axis=0)
covariate_df = df[['Weight', 'Age', 'Sex']]

shap.summary_plot(covariate_shap_values, covariate_df, show=False, sort=False)
ax = plt.gca()
ax.set_ylabel("Covariate")
ax.set_xlim(-1.25, 1.25)
ax.set_xticks(np.arange(-1.25, 1.26, 0.25))
plt.savefig('Shap/shap_summary_plot.pdf', bbox_inches='tight')
plt.close()



