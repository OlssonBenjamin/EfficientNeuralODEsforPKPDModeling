##############################################################################
# Imports
##############################################################################

import torchdiffeq as tdq                     # Main ODE solver package
from torchdiffeq import odeint               # Explicit import for convenience
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import copy


##############################################################################
# 1.  SWAG utilities
##############################################################################

class SWAGTracker:
    """
    Collects parameter “snapshots” during training and, at the end, builds a
    Gaussian approximation of the posterior (mean & full covariance).
    This is the classical SWA-Gaussian (SWAG) approach by Maddox et al. (2019).
    """

    def __init__(self):
        self.snapshots = []       # list[dict[str, Tensor]] – stored state_dicts
        self.mean_param = None    # 1-D tensor of concatenated means
        self.cov_param  = None    # 2-D tensor, full covariance matrix

    # --------------------------------------------------------------------- #
    def collect_model(self, model: nn.Module):
        """Deep-copy and store the current parameters of `model`."""
        self.snapshots.append(copy.deepcopy(model.state_dict()))

    # --------------------------------------------------------------------- #
    def _flatten_state(self, state_dict: dict) -> torch.Tensor:
        """
        Helper: turn an ordered state_dict into a single 1-D vector
        (preserves parameter order).
        """
        flat_chunks = [p.view(-1) for p in state_dict.values()]
        return torch.cat(flat_chunks)

    # --------------------------------------------------------------------- #
    def finalize(self):
        """
        After training, call this once to compute
        • self.mean_param  – concatenated mean of parameters
        • self.cov_param   – empirical covariance across snapshots
        """
        if len(self.snapshots) == 0:
            raise ValueError("No snapshots collected for SWAG.")

        # Build matrix  (num_snapshots × num_params)
        param_matrix = torch.stack(
            [self._flatten_state(s) for s in self.snapshots], dim=0
        )

        # Empirical mean
        self.mean_param = param_matrix.mean(dim=0)               # shape [P]
        # Empirical (unbiased) covariance  Σ = 1/(N-1) XᵀX
        centered = param_matrix - self.mean_param.unsqueeze(0)   # shape [N, P]
        self.cov_param = centered.t() @ centered / (len(self.snapshots) - 1)

        return self.mean_param, self.cov_param

    # --------------------------------------------------------------------- #
    def sample(self) -> torch.Tensor:
        """
        Draw one sample θ ~ 𝓝(mean, cov). Returns a flat 1-D tensor.
        Warning: Cholesky on a huge covariance is expensive; consider low-rank.
        """
        if self.mean_param is None or self.cov_param is None:
            raise ValueError("Call finalize() before sampling.")

        eps = torch.randn_like(self.mean_param)     # Standard normal
        L   = torch.linalg.cholesky(self.cov_param) # Covariance factor
        return self.mean_param + L @ eps            # θ_sample


##############################################################################
# 2.  Dataset helpers
##############################################################################

class PatientDataset(Dataset):
    """
    Thin wrapper so we can feed a (N_patients × 7 × T) numpy array to a
    DataLoader. Indexing returns a single patient tensor of shape [7, T].
    """

    def __init__(self, data: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


##############################################################################
# 3.  Neural ODE infrastructure
##############################################################################

class NeuralODE(nn.Module):
    """
    Generic Neural ODE wrapper – calls torchdiffeq.odeint using Euler steps.
    The underlying RHS (ode_func) is provided during construction.
    """

    def __init__(self, ode_func: nn.Module, step_size: float = 0.1):
        super().__init__()
        self.ode_func  = ode_func
        self.step_size = step_size

    # t is a 1-D tensor of times; input is batch × state_dim
    def forward(self, input, t):
        return odeint(
            self.ode_func,
            input,
            t,
            method="euler",
            options=dict(step_size=self.step_size),
        )


##############################################################################
# 4.  Parameterised RHS of the ODE (PK/PD-style)
##############################################################################

class TrainODEFunc(nn.Module):
    """
    ODE right-hand side that switches between an 'increasing' and a
    'decreasing' regime depending on how long it has been since the last dose.
    """

    def __init__(self, bottleneck: int, peak_dose: float = 4.0):
        super().__init__()

        # “Active uptake” network (after a new dose)
        self.increasing_net = nn.Sequential(
            nn.Linear(bottleneck + 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),        # guarantees non-negative output
        )

        # “Clearance” network (long after a dose)
        self.decreasing_net = nn.Sequential(
            nn.Linear(bottleneck + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.bottleneck = bottleneck
        self.peak_dose  = peak_dose         # threshold that defines regime

    # ------------------------------------------------------------------ #
    def forward(self, t: torch.Tensor, y: torch.Tensor):
        """
        Parameters
        ----------
        t : scalar tensor                           – the current time
        y : (batch × state) tensor with fields
            • y[:,0]   – current concentration  C(t)
            • y[:,1:1+K]          – “steering variables” (covariates)
            • y[:,1+K:-1]         – dosing schedule (vector of times)
            • y[:,-1]             – normalised dose size

        Returns
        -------
        dzdt : tensor with same shape as y (padding zeros for non-C entries)
        """

        # --- Unpack convenient aliases ---------------------------------- #
        current_conc      = y[:, 0].unsqueeze(1)                 # C(t)
        steering_vars     = y[:, 1 : 1 + self.bottleneck]        # covariates
        dosing_times      = y[:, 1 + self.bottleneck : -1]       # schedule
        dose_size         = y[:, -1].unsqueeze(1)                # scalar
        batch, n_doses    = dosing_times.shape

        # --- How long since last administered dose? -------------------- #
        t_expanded        = t.expand(batch, n_doses)             # (B × n_doses)
        mask              = dosing_times <= t_expanded           # doses given
        last_dose_time    = (dosing_times * mask).max(dim=1, keepdim=True)[0]
        time_since_last   = t - last_dose_time                   # Δt

        # --- Choose network depending on regime ------------------------ #
        if (time_since_last >= self.peak_dose).any():
            # Clearance phase
            ode_in  = torch.cat([steering_vars,
                                 time_since_last,
                                 current_conc], dim=1)
            dCdt    = -self.decreasing_net(ode_in)               # negative
        else:
            # Uptake phase
            ode_in  = torch.cat([steering_vars,
                                 time_since_last,
                                 current_conc,
                                 dose_size], dim=1)
            dCdt    =  self.increasing_net(ode_in)

        # Pad derivative so its shape matches `y`
        zeros_steer  = torch.zeros_like(steering_vars)
        zeros_sched  = torch.zeros_like(dosing_times)
        zeros_dose   = torch.zeros_like(dose_size)
        dzdt         = torch.cat([dCdt,
                                  zeros_steer,
                                  zeros_sched,
                                  zeros_dose], dim=1)
        return dzdt


##############################################################################
# 5.  Full model that uses the ODEFunc
##############################################################################

class TrainODE(nn.Module):
    """
    Combines:
        • patient_encoding  – maps static covariates to a low-dim embedding
        • Neural ODE        – integrates concentration trajectory
        • solve(…)          – simulate entire trajectory
        • forward(…)        – sampled-time FM loss (flow-matching)
    """

    def __init__(self,
                 num_variables: int,
                 bottleneck: int,
                 zero_dose: float):
        super().__init__()

        # Static covariate encoder ------------------------------------- #
        self.patient_encoding = nn.Sequential(
            nn.Linear(num_variables + 1, 32),
            nn.SiLU(),
            nn.Linear(32, bottleneck),
        )

        # ODE machinery ----------------------------------------------- #
        self.ode     = TrainODEFunc(bottleneck)
        self.node    = NeuralODE(self.ode)

        # Noise scale used when sampling “interpolants” for FM training
        self.std         = 7.5
        self.zero_dose   = zero_dose

    # ------------------------------------------------------------------ #
    def forward(self,
                covariates,          # (B × num_variables)
                dosing_times,        # (B × n_doses)
                predict_times,       # 1-D tensor of length T
                trajectories):       # (B × T) – true CP values
        """
        Implements the “flow-matching” loss of Dolz et al. (2023):
        pick random intermediate time points, interpolate the data, and
        compare analytical dC/dt with RHS network output.
        """

        # 1. Encode static patient covariates -------------------------- #
        x = self.patient_encoding(covariates)            # (B × bottleneck)

        # 2. Create fine-grained time grid (0.1-h resolution) ---------- #
        fine_times = torch.linspace(predict_times[0],
                                    predict_times[-1],
                                    steps=int(predict_times[-1] * 10 + 1))

        fine_times = fine_times[~torch.isin(fine_times, predict_times[-1])]

        # 3. Sample K random time points from this fine grid ----------- #
        K                = 1
        rnd_idx          = torch.randint(0, len(fine_times), (K,))
        random_times     = fine_times[rnd_idx]                 # length K
        idx_right        = torch.searchsorted(predict_times, random_times)
        idx_left         = idx_right - 1

        # If sampled exactly on existing t_i, tweak indices ------------- #
        on_grid          = torch.isin(random_times, predict_times)
        idx_left[on_grid]  = idx_right[on_grid]
        idx_right[on_grid] = idx_right[on_grid] + 1

        # 4. Bilinear interpolation of CP values at those times -------- #
        t_left   = predict_times[idx_left]
        t_right  = predict_times[idx_right]
        alpha    = ((random_times - t_left) / (t_right - t_left)).unsqueeze(0)

        batch = trajectories.size(0)
        interp_left  = torch.gather(trajectories, 1,
                                    idx_left.unsqueeze(0).expand(batch, -1))
        interp_right = torch.gather(trajectories, 1,
                                    idx_right.unsqueeze(0).expand(batch, -1))
        interpolants = interp_left + alpha * (interp_right - interp_left)

        # 5. Add small noise to stabilise training --------------------- #
        if covariates[0, -1] == self.zero_dose:
            interpolants = torch.normal(interpolants, 0.5)
        else:
            interpolants = torch.normal(interpolants, self.std)

        # 6. Evaluate RHS dC/dt for each sampled time ------------------ #
        dzdt_list = []
        for k in range(K):
            t_k     = random_times[k]
            interp  = interpolants[:, k].unsqueeze(1)

            encoding_k = torch.cat([interp,
                                    x,
                                    dosing_times,
                                    covariates[:, -1].unsqueeze(1)], dim=1)
            # ODEFunc returns batch × state_dim; keep first column dC/dt
            dzdt_k = self.ode(t_k, encoding_k)[:, 0]
            dzdt_list.append(dzdt_k)

        dzdt_all = torch.stack(dzdt_list, dim=1)   # (B × K)

        return dzdt_all, random_times, idx_right, t_right, interpolants

    # ------------------------------------------------------------------ #
    def decay_std(self, factor: float = 0.99):
        """Slowly anneal noise σ after each epoch."""
        if self.std >= 1.0:
            self.std *= factor

    # ------------------------------------------------------------------ #
    def solve(self,
              covariates,
              dosing_times,
              predict_times):
        """
        Integrate the ODE across the full time grid and return
        non-negative concentration profile C(t) (ReLU applied).
        """
        x      = self.patient_encoding(covariates)
        batch  = x.size(0)
        y0     = torch.zeros(batch, 1)                      # initial C(0)=0

        # Concatenate all state entries required by ODEFunc
        encoding0 = torch.cat([y0, x, dosing_times,
                               covariates[:, -1].unsqueeze(1)], dim=1)

        latent_ts = self.node(encoding0, predict_times)     # shape (T × B × D)
        C_ts      = nn.functional.relu(latent_ts[:, :, 0])  # keep C(t) ≥ 0

        self.decay_std()                                    # anneal noise σ
        return C_ts


##############################################################################
# 6.  Weight initialisation helper
##############################################################################

def initialize_weights(m: nn.Module):
    """Xavier initialisation for Linear layers, smaller gain (0.25)."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.25)
        m.bias.data.fill_(0.0)


##############################################################################
# 7.  Data-loading utilities (CSV → numpy → torch)
##############################################################################
#
#   Both create a 3-D tensor   shape = (#patients × 7 × #samples)
#   with channels:
#     0) time
#     1) CP
#     2) weight
#     3) age
#     4) sex
#     5) dose size (z-scored)
#     6) regimen index (0,1,2)
##############################################################################

# Due to length, only the first loader is fully commented; the validation
# loader is analogous.

def load_data(data_file, covariate_file,
              dose_size_norm,   # already z-scored
              dose_regime_index):
    """
    Read PK/PD CSV files, merge with covariates, build unified numpy array.
    """

    # ---- 1. per-patient CP/time/response ---------------------------- #
    df_pkpd         = pd.read_csv(data_file)
    cp_cols         = df_pkpd[['ID', 'time', 'CP', 'RESPONSE']]
    grouped_patients= cp_cols.groupby('ID')

    patient_pkpd = {}
    for pid, grp in grouped_patients:
        patient_pkpd[pid] = dict(time     = grp['time'].values,
                                 CP       = grp['CP'].values,
                                 RESPONSE = grp['RESPONSE'].values)

    # ---- 2. per-patient static covariates --------------------------- #
    df_cov      = pd.read_csv(covariate_file)
    cov_cols    = df_cov[['ID', 'WT', 'AGE', 'SEX']]
    grouped_cov = cov_cols.groupby('ID')

    patient_cov = {}
    for pid, grp in grouped_cov:
        patient_cov[pid] = dict(WT  = grp['WT'].values,
                                AGE = grp['AGE'].values,
                                SEX = grp['SEX'].values)

    # ---- 3. Build final 3-D array  (#patients × 7 × T) -------------- #
    T           = len(patient_pkpd[1]['time'])
    all_data    = np.zeros((75, 7, T))

    for i in range(1, 76):
        all_data[i-1, 0, :] = patient_pkpd[i]['time']
        all_data[i-1, 1, :] = patient_pkpd[i]['CP']
        all_data[i-1, 2, :] = patient_cov[i]['WT']
        all_data[i-1, 3, :] = patient_cov[i]['AGE']
        all_data[i-1, 4, :] = patient_cov[i]['SEX']
        all_data[i-1, 5, :] = dose_size_norm
        all_data[i-1, 6, :] = dose_regime_index

    # ---- 4. Sub-sample time points (sampling strategy hard-coded) --- #
    idx = np.zeros(35, dtype=int)
    idx[1]  = 41
    idx[2]  = 41 + 200
    k = 3
    while k < 35:
        idx[k]   = idx[k-1] + 40
        idx[k+1] = idx[k]   + 200
        k += 2
    all_data = all_data[:, :, idx]

    # Torch time tensor and boolean mask for convenience
    time_tensor = torch.tensor(all_data[0, 0, :], dtype=torch.float32)
    mask        = torch.isin(time_tensor, time_tensor)   # all True here
    return time_tensor, mask, all_data

def load_validation_data(data_file, covariate_file,
              dose_size_norm,   # already z-scored
              dose_regime_index):
     # ---- 1. per-patient CP/time/response ---------------------------- #
    df_pkpd         = pd.read_csv(data_file)
    cp_cols         = df_pkpd[['ID', 'time', 'CP', 'RESPONSE']]
    grouped_patients= cp_cols.groupby('ID')

    patient_pkpd = {}
    for pid, grp in grouped_patients:
        patient_pkpd[pid] = dict(time     = grp['time'].values,
                                 CP       = grp['CP'].values,
                                 RESPONSE = grp['RESPONSE'].values)

    # ---- 2. per-patient static covariates --------------------------- #
    df_cov      = pd.read_csv(covariate_file)
    cov_cols    = df_cov[['ID', 'WT', 'AGE', 'SEX']]
    grouped_cov = cov_cols.groupby('ID')

    patient_cov = {}
    for pid, grp in grouped_cov:
        patient_cov[pid] = dict(WT  = grp['WT'].values,
                                AGE = grp['AGE'].values,
                                SEX = grp['SEX'].values)

    # ---- 3. Build final 3-D array  (#patients × 7 × T) -------------- #
    T           = len(patient_pkpd[1]['time'])
    all_data    = np.zeros((25, 7, T))

    for i in range(76, 101):
        all_data[i-76, 0, :] = patient_pkpd[i]['time']
        all_data[i-76, 1, :] = patient_pkpd[i]['CP']
        all_data[i-76, 2, :] = patient_cov[i]['WT']
        all_data[i-76, 3, :] = patient_cov[i]['AGE']
        all_data[i-76, 4, :] = patient_cov[i]['SEX']
        all_data[i-76, 5, :] = dose_size_norm
        all_data[i-76, 6, :] = dose_regime_index

    # ---- 4. Sub-sample time points (sampling strategy hard-coded) --- #
    idx = np.zeros(35, dtype=int)
    idx[1]  = 41
    idx[2]  = 41 + 200
    k = 3
    while k < 35:
        idx[k]   = idx[k-1] + 40
        idx[k+1] = idx[k]   + 200
        k += 2
    all_data = all_data[:, :, idx]

    # Torch time tensor and boolean mask for convenience
    time_tensor = torch.tensor(all_data[0, 0, :], dtype=torch.float32)
    mask        = torch.isin(time_tensor, time_tensor)   # all True here
    return time_tensor, mask, all_data


##############################################################################
# 8.  Training set-up
##############################################################################

# ---- Normalise dose sizes ---------------------------------------------- #
dose_sizes   = np.array([100, 50, 0])
dose_size_1  = (100 - dose_sizes.mean()) / dose_sizes.std()
dose_size_2  = ( 50 - dose_sizes.mean()) / dose_sizes.std()
dose_size_3  = (  0 - dose_sizes.mean()) / dose_sizes.std()

# ---- Load 3 regimens (train + validation) ------------------------------ #
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

# ---- Dosing times tensors --------------------------------------------- #
dose_times_1 = torch.tensor([0, 24, 48, 72, 96], dtype=torch.float32)
dose_times_2 = torch.tensor([0, 24, 48, 72, 96, 120, 144, 168, 192, 216],
                            dtype=torch.float32)

# ---- Trainers / loaders ------------------------------------------------ #
batch_size = 8
train_loaders = [
    DataLoader(PatientDataset(all_patient_data_1), batch_size, shuffle=True),
    DataLoader(PatientDataset(all_patient_data_2), batch_size, shuffle=True),
    DataLoader(PatientDataset(all_patient_data_3), batch_size, shuffle=True),
]

val_batch_size = 25
val_loaders = [
    DataLoader(PatientDataset(val_patient_data_1), val_batch_size, shuffle=False),
    DataLoader(PatientDataset(val_patient_data_2), val_batch_size, shuffle=False),
    DataLoader(PatientDataset(val_patient_data_3), val_batch_size, shuffle=False),
]

##############################################################################
# 9.  Model instantiation
##############################################################################

bottleneck = 4
model      = TrainODE(num_variables=3, bottleneck=bottleneck,
                      zero_dose=dose_size_3)
model.apply(initialize_weights)

optimizer   = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
epochs      = 2001
loss_fn     = nn.MSELoss()
loss_fn_sum = nn.MSELoss(reduction="sum")  # for validation

swag_tracker = SWAGTracker()
best_val_loss = float("inf")

##############################################################################
# 10. Training loop (flow-matching)
##############################################################################

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0

    # ------------------------------------------------------------- #
    for loader in train_loaders:
        for batch in loader:
            optimizer.zero_grad()

            # Unpack batch  – shape [B, 7, T]
            covariates = batch[:, 2:6, 0].float()
            cp_traj    = batch[:, 1, :].float()

            # Build dosing-time matrix per regimen ------------------- #
            regimen = int(batch[0, 6, 0].item())
            if   regimen == 0:
                dose_times = dose_times_1.expand(batch.size(0), -1)
            elif regimen == 1:
                dose_times = dose_times_2.expand(batch.size(0), -1)
            else:  # random #doses between 0 and 10
                n = random.randint(0, len(dose_times_2))
                dose_times_rand = dose_times_2[:n] if n else torch.tensor([], dtype=torch.float32)
                dose_times = dose_times_rand.expand(batch.size(0), -1) if n else None

            # Evaluate flow-matching loss --------------------------- #
            dzdt, _, idx_right, t_right, interp = model(
                covariates, dose_times, full_time_tensor, cp_traj
            )
            # Target derivative via finite difference
            tgt = (cp_traj[:, idx_right] - interp) / (t_right - _)
            loss = loss_fn(dzdt, tgt.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

    # --- Optionally store SWAG snapshot --------------------------- #
    if epoch >= 1000 and epoch % 10 == 0:
        swag_tracker.collect_model(model)

    # ------------------------------------------------------------- #
    # Validation (every 1000 epochs)                                #
    # ------------------------------------------------------------- #
    if epoch % 1000 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vloader in val_loaders:
                for batch in vloader:
                    covariates = batch[:, 2:6, 0].float()
                    cp_traj    = batch[:, 1, :].float()
                    regimen    = int(batch[0, 6, 0].item())
                    dose_times = (dose_times_1 if regimen == 0 else
                                  dose_times_2).expand(batch.size(0), -1)

                    # Simulate complete trajectory
                    cp_hat = model.solve(covariates, dose_times,
                                         full_time_tensor).transpose(0, 1)
                    val_loss += loss_fn_sum(cp_hat, cp_traj) / (25 * 3 * cp_traj.size(-1))

        # Save model if improved
        if val_loss < best_val_loss:
            torch.save(model, 'flow_matching_model_swag')
            best_val_loss = val_loss

        print(f"[{epoch:4d}]  TrainLoss = {epoch_train_loss:.4f} | "
              f"ValLoss = {val_loss:.4f}")
        


# ------------------------------------------------------------------------- #
# 11. Finish SWAG                                                           #
# ------------------------------------------------------------------------- #
""" mean_vec, cov_mat = swag_tracker.finalize()
torch.save(mean_vec, 'swag_mean.pth')
torch.save(cov_mat, 'swag_cov.pth') """
                    
