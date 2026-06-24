from torchdiffeq import odeint
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import shap

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
           nn.Linear(bottleneck + 4, 64),
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
        num_dose = num_dose.expand([y.size(0), 1])
        masked_dosing_times = dosing_times * mask 
        last_dose = t - masked_dosing_times.max(dim=1, keepdim=True)[0]
        if (last_dose >= self.peak_dose).any():
            ode_input = torch.cat([steering_variables], dim=1)
            dzdt = -(self.decreasing_net(ode_input)) * current_concentration * min(last_dose[0] - self.peak_dose, 1)
        else:
            ode_input = torch.cat([steering_variables, dose_size, last_dose, current_concentration, num_dose], dim=1)
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
       self.std = 7.5
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

        return dzdt, random_times, bigger_idx, smaller_idx, bigger_times, smaller_times, interpolants, last_dose
    

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
    def __init__(self, patient_encoding, increasing_net, decreasing_net, zero_dose):
       super(TestNetwork, self).__init__()
       self.patient_encoding = patient_encoding
       self.increasing_net = increasing_net
       self.decreasing_net = decreasing_net
       self.peak_dose = 4
       self.zero_dose = zero_dose

    def forward(self, input):
        covariates = input[:, :3]
        dose_size = input[:, 3].unsqueeze(-1)
        last_dose = input[:, 4].unsqueeze(-1)
        current_concentration = input[:, 5].unsqueeze(-1)
        num_dose = input[:, 6].unsqueeze(-1)
        steering_variables = self.patient_encoding(covariates)
        dzdt = torch.zeros((input.size(0), 1))
        for i in range(input.size(0)):
            if not (dose_size[i][0] == self.zero_dose):
                ode_input = torch.cat([steering_variables[i].unsqueeze(0), dose_size[i].unsqueeze(0), last_dose[i].unsqueeze(0), current_concentration[i].unsqueeze(0), num_dose[i].unsqueeze(0)], dim=1)
                dzdt[i][0] = self.increasing_net(ode_input)
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
num_dose = torch.tensor(0.0, dtype=torch.float32).expand([225, 1])

input_1 = all_patient_data_1[:, 2:6, 0]
input_2 = all_patient_data_2[:, 2:6, 0]
input_3 = all_patient_data_3[:, 2:6, 0]
input = np.vstack([input_1, input_2, input_3])

input = torch.tensor(input)
input_tensor = torch.cat([input, last_dose, current_concentration, num_dose], dim=1).to(torch.float32)
input = input_tensor.numpy()

column_names = ['Weight', 'Age', 'Sex', 'Dose Size', 'Last Dose', 'Current Concentration', 'Number of Doses Taken']

df = pd.DataFrame(input, columns=column_names)

model = torch.load('flow_matching_model_0')

patient_encoding_network = model.patient_encoding
increasing_net = model.ode.increasing_net
decreasing_net = model.ode.decreasing_net
shap_model = TestNetwork(patient_encoding_network, increasing_net, decreasing_net, dose_size_3)

explainer = shap.GradientExplainer(shap_model, input_tensor)
shap_values = explainer.shap_values(input_tensor)[:, :, 0]
shap.plots.violin(shap_values)
shap.summary_plot(shap_values, df)
shap.dependence_plot("Dose Size", shap_values, df)
shap.dependence_plot("Age", shap_values, df)
shap.dependence_plot("Weight", shap_values, df)
shap.dependence_plot("Sex", shap_values, df)



