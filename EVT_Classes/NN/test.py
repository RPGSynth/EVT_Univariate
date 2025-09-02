# ==============================================================================
# SECTION 0 — PYTORCH SETUP (replace JAX/Flax for sections 1–4)
# ==============================================================================
import os
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PyTorch] Using device: {device}")
torch.manual_seed(0)
np.random.seed(0)

DIST_EPSILON = 1e-8  # numeric stability


# ==============================================================================
# SECTION 1 — DATA PREPARATION & UTILITIES (NumPy + PyTorch)
# ==============================================================================
def get_simulated_data(n_samples=1000, seed=123, **kwargs):
    """
    Generate 1D non-stationary data with heteroscedastic noise.
    Returns:
      locs: (N,1) locations (same as features here)
      X_features: (N,1) features
      y: (N,) targets
    """
    rng = np.random.default_rng(seed)
    config = {
        'x_minval': -2 * np.pi, 'x_maxval': 2 * np.pi, 'curve_type': 'sin',
        'amplitude': 1.5, 'frequency': 1.0, 'phase': 0.0, 'vertical_offset': 0.5,
        'x_slope_coeff': 1.0, 'noise_y_std': 0.3, 'noise_beta0_std': 0.5,
        'noise_beta1_std': 0.05, 'noise_type': 'wavy'
    }
    config.update(kwargs)

    X_features = np.linspace(config['x_minval'], config['x_maxval'], n_samples).reshape(-1, 1)
    locs = X_features  # for plotting consistency

    main_curve = config['amplitude'] * np.sin(config['frequency'] * locs.flatten() + config['phase'])
    beta0_noise = rng.normal(0.0, config['noise_beta0_std'], size=n_samples)
    beta0_values = config['vertical_offset'] + main_curve + beta0_noise

    beta1_noise = rng.normal(0.0, config['noise_beta1_std'], size=n_samples)
    beta1_values = config['x_slope_coeff'] + beta1_noise

    y_deterministic = beta0_values + beta1_values * locs.flatten()

    # heteroscedastic noise
    wave = np.sin(2.5 * locs.flatten() + 0.3) + np.sin(6.3 * locs.flatten() + 1.8)
    pattern = np.abs(wave) + 0.2
    pattern /= np.mean(pattern)
    dynamic_std = (config['noise_y_std'] * (pattern ** 1.5)).astype(np.float64)
    y_noise = rng.normal(0.0, 1.0, size=n_samples) * dynamic_std

    y = (y_deterministic + y_noise).astype(np.float32)
    return locs.astype(np.float32), X_features.astype(np.float32), y

# --- Data split & scaling (NumPy) ---
locs, X, y = get_simulated_data(n_samples=10000, noise_y_std=0.8, x_slope_coeff=0.9)
locs_train_val, locs_test, X_train_val, X_test, y_train_val, y_test = train_test_split(
    locs, X, y, test_size=0.20, random_state=42
)
locs_train, locs_val, X_train, X_val, y_train, y_val = train_test_split(
    locs_train_val, X_train_val, y_train_val, test_size=0.25, random_state=42
)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)

# Torch tensors on device
X_train_t = torch.from_numpy(X_train_scaled).float().to(device)
y_train_t = torch.from_numpy(y_train).float().to(device)
X_val_t   = torch.from_numpy(X_val_scaled).float().to(device)
y_val_t   = torch.from_numpy(y_val).float().to(device)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)


# ==============================================================================
# SECTION 2 — TEACHER MODEL (GlobalMLP) IN PYTORCH
# ==============================================================================
class GlobalMLP(tnn.Module):
    """
    MLP with two heads: μ(x) and log σ²(x).
    Input:  (B, D)
    Output: mu: (B,), log_sigma2: (B,)
    """
    def __init__(self, in_dim, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        last = in_dim
        for i, h in enumerate(hidden_dims):
            layers += [tnn.Linear(last, h), tnn.PReLU()]
            last = h
        self.backbone = tnn.Sequential(*layers)
        self.mu_head = tnn.Linear(last, 1)
        self.logsigma2_head = tnn.Linear(last, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)          # (B,)
        log_sigma2 = self.logsigma2_head(h).squeeze(-1)  # (B,)
        return mu, log_sigma2

def nll_loss_from_logits(mu_pred, log_sigma2_pred, targets):
    """
    Gaussian NLL where model predicts log σ² directly.
    Shapes: (B,), (B,), (B,)
    """
    log_sigma2_pred = torch.clamp(log_sigma2_pred, min=-15.0, max=15.0)
    sigma2 = torch.exp(log_sigma2_pred) + DIST_EPSILON
    return torch.mean(0.5 * (np.log(2 * np.pi) + torch.log(sigma2) + (targets - mu_pred)**2 / sigma2))


# --- init model/optim/dataloaders ---
model = GlobalMLP(in_dim=X_train_t.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=False)

# ==============================================================================
# SECTION 2.1 — TRAIN LOOP (teacher)
# ==============================================================================
epochs, patience = 500, 50
best_val_loss, patience_counter = float('inf'), 0

# snapshot best weights
best_state_dict = None

model.train()
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        mu_pred, log_sigma2_pred = model(xb)
        loss = nll_loss_from_logits(mu_pred, log_sigma2_pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)

    # simple validation on full val set (small)
    model.eval()
    with torch.no_grad():
        mu_val, log_sigma2_val = model(X_val_t)
        val_loss = nll_loss_from_logits(mu_val, log_sigma2_val, y_val_t).item()

    if (epoch + 1) % 10 == 0:
        print(f"Teacher | Epoch {epoch+1:03d} | Train NLL: {avg_train_loss:.6f} | Val NLL: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Teacher training stopped early at epoch {epoch+1}.")
        break

# restore best
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
model.eval()


# ==============================================================================
# SECTION 3 — DIAGNOSTIC PLOTS FOR THE TEACHER MODEL
# ==============================================================================
print("\n--- Generating Diagnostic Plots for Trained Teacher (PyTorch) ---")
plt.style.use('seaborn-v0_8-whitegrid')

X_all_scaled = scaler_X.transform(X).astype(np.float32)
X_all_t = torch.from_numpy(X_all_scaled).to(device)

with torch.no_grad():
    mu_all_t, log_sigma2_all_t = model(X_all_t)
    log_sigma2_all_t = torch.clamp(log_sigma2_all_t, min=-15.0, max=15.0)
    sigma_all_t = torch.exp(0.5 * log_sigma2_all_t)

mu_all = mu_all_t.detach().cpu().numpy()
sigma_all = sigma_all_t.detach().cpu().numpy()

plt.figure(figsize=(12, 7))
sorted_idx = np.argsort(X[:, 0])
plt.scatter(X[:, 0], y, label='Data Points', alpha=0.3, s=20, color='gray')
plt.plot(X[sorted_idx, 0], mu_all[sorted_idx], label='Predicted Mean (μ)', color='firebrick', linewidth=2.5)
plt.fill_between(X[sorted_idx, 0],
                 (mu_all - 2 * sigma_all)[sorted_idx],
                 (mu_all + 2 * sigma_all)[sorted_idx],
                 color='firebrick', alpha=0.2, label='Uncertainty (±2σ)')
plt.title('Teacher Model (GlobalMLP / PyTorch): Predictions with Uncertainty', fontsize=16)
plt.xlabel('Feature X (Location)'); plt.ylabel('Target y'); plt.legend()
plt.tight_layout(); plt.show()


# ==============================================================================
# SECTION 4 — GENERATE TEACHER TARGETS (μ, σ) FOR TRAIN SET
# ==============================================================================
print("\n--- Generating Target Parameters (μ, σ) from Trained Teacher ---")
with torch.no_grad():
    mu_teacher_t, log_sigma2_teacher_t = model(X_train_t)
    sigma_teacher_t = torch.exp(0.5 * torch.clamp(log_sigma2_teacher_t, min=-15.0, max=15.0))

mu_teacher_train = mu_teacher_t.detach().cpu().numpy()     # (N_train,)
sigma_teacher_train = sigma_teacher_t.detach().cpu().numpy()  # (N_train,)
print(f"Generated teacher's μ and σ for {mu_teacher_train.shape[0]} training points.")


# ==============================================================================
# SECTION 5 — UNIFORM RANDOM PAIR PICKING (no kNN) + distance scaling  [PyTorch]
# ==============================================================================
import torch 

locs_train_t = torch.from_numpy(locs_train.astype(np.float32)).to(device)   # (N,1)
y_train_t    = torch.from_numpy(y_train.astype(np.float32)).to(device)      # (N,)

# Teacher outputs from Section 4 (NumPy) → Torch
mu_teacher_t    = torch.from_numpy(mu_teacher_train.astype(np.float32)).to(device)      # (N,)
sigma_teacher_t = torch.from_numpy(sigma_teacher_train.astype(np.float32)).to(device)   # (N,)


K_SAMPLES = 500   # number of random partners per query (was K_NEIGHBORS)

@torch.inference_mode()
def estimate_global_dist_scale(locs: torch.Tensor, num_pairs: int = 200_000) -> torch.Tensor:
    """
    Robust global distance scale ~ median distance over a large random set of pairs.
    Excludes self-pairs via the 'skip self' trick.
    """
    N = locs.shape[0]
    M = min(num_pairs, max(1, N * (N - 1)))
    i = torch.randint(0, N, (M,), device=device)
    j = torch.randint(0, N - 1, (M,), device=device)
    j = j + (j >= i).long()  # shift to avoid j == i per element
    d = torch.linalg.norm(locs[i] - locs[j], dim=-1)
    return torch.median(d)

def sample_uniform_partners(batch_query_idx: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """
    For each query index in batch_query_idx (B,), sample K distinct indices
    uniformly from {0..N-1} \ {q}, without replacement.
    Uses a tiny per-row randperm loop (fast at typical batch sizes).
    """
    assert K <= max(1, N - 1), f"K={K} must be <= N-1 (N={N})"
    device = batch_query_idx.device
    B = batch_query_idx.shape[0]
    partners = torch.empty((B, K), dtype=torch.long, device=device)
    for b in range(B):
        q = int(batch_query_idx[b].item())
        # draw from 0..N-2, then map to 0..N-1 with "skip-self" shift
        idx = torch.randperm(N - 1, device=device)[:K]
        partners[b] = idx + (idx >= q)
    return partners

def rowwise_dists(locs: torch.Tensor, query_idx: torch.Tensor, partner_idx: torch.Tensor) -> torch.Tensor:
    """
    Compute ||loc[q] - loc[i]|| for each (q,i) pair in a batch.
    locs: (N,D), query_idx: (B,), partner_idx: (B,K) -> (B,K)
    """
    qloc = locs.index_select(0, query_idx).unsqueeze(1)     # (B,1,D)
    ploc = locs.index_select(0, partner_idx.view(-1)).view(partner_idx.shape + (locs.shape[1],))  # (B,K,D)
    return torch.linalg.norm(ploc - qloc, dim=-1)           # (B,K)

with torch.inference_mode():
    dist_scale = estimate_global_dist_scale(locs_train_t)  # scalar on device
print(f"[Uniform] Distance scaling median = {float(dist_scale):.6f}")


# ==============================================================================
# SECTION 5.1 — ESS HELPERS (no normalization needed)  [PyTorch]
# ==============================================================================
EPS = 1e-8

@torch.inference_mode()
def ess_rows_from_raw(W: torch.Tensor) -> torch.Tensor:
    """
    Kish ESS per row from unnormalized weights W: ESS = (sum w)^2 / sum(w^2).
    W: (B,K) -> (B,)
    """
    row_sum = torch.sum(W, dim=1) + EPS
    return (row_sum * row_sum) / (torch.sum(W * W, dim=1) + EPS)


# ==============================================================================
# SECTION 6 — DYNAMIC WEIGHT NET (learn w(d)) + TRAIN LOOP (ESS early stop)  [PyTorch]
# ==============================================================================

class DynamicWeightNet(tnn.Module):
    """
    Kernel network: takes scaled distance d̃ and outputs raw weight w(d̃) ∈ (0,1).
    Input:  (B,1) distances
    Output: (B,1) weights in (0,1)
    """
    def __init__(self, hidden_dims=(32, 16), bias_init=0.8):
        super().__init__()
        layers = []
        last = 1
        for h in hidden_dims:
            layers += [tnn.Linear(last, h), tnn.ReLU()]
            last = h
        self.backbone = tnn.Sequential(*layers)
        self.out = tnn.Linear(last, 1)
        with torch.no_grad():
            self.out.bias.fill_(bias_init)  # sigmoid(0.8) ~ 0.69 at init

    def forward(self, x):
        h = self.backbone(x)
        logits = self.out(h)
        return torch.sigmoid(logits)  # (0,1)

def rowsum_normalize(W: torch.Tensor) -> torch.Tensor:
    s = torch.sum(W, dim=1, keepdim=True).clamp_min(1e-8)
    return W / s

def query_weighted_nll_loss(w_raw: torch.Tensor,
                            mu_n: torch.Tensor,
                            sigma_n: torch.Tensor,
                            y_q: torch.Tensor) -> torch.Tensor:
    """
    Loss per query: Σ_i w_norm(q,i) * NLL(y_q | μ_i, σ_i); mean over batch.
    Shapes:
      w_raw:  (B,K)
      mu_n:   (B,K)
      sigma_n:(B,K)      # σ (std), not variance
      y_q:    (B,)
    """
    B, K = w_raw.shape
    w_norm = rowsum_normalize(w_raw)                # train-time normalization (shape learning)
    yq = y_q.view(B, 1)
    safe_sigma = sigma_n + 1e-8
    nll_pair = 0.5 * (np.log(2 * np.pi) + 2.0 * torch.log(safe_sigma) +
                      ((yq - mu_n) ** 2) / (safe_sigma ** 2))          # (B,K)
    loss_per_q = torch.sum(w_norm * nll_pair, dim=1)                    # (B,)
    return torch.mean(loss_per_q)

dynamic_net = DynamicWeightNet().to(device)
dyn_optim   = torch.optim.Adam(dynamic_net.parameters(), lr=1e-4)

# ESS-based early stopping config
ESS_MIN_THRESHOLD = 333   # choose relative to K_SAMPLES (e.g., 40–150 for K=500)
ESS_SAMPLE_SIZE   = 100   # number of query rows to estimate ESS each epoch

# --- dynamic kernel training loop with UNIFORM pairs ---
epochs, batch_size, patience = 500, 512, 50
best_train_loss, patience_counter = float('inf'), 0
best_dyn_state = {k: v.detach().cpu().clone() for k, v in dynamic_net.state_dict().items()}

print(f"\nTraining dynamic model (uniform pairs) for {epochs} epochs...")

N = locs_train_t.shape[0]
for epoch in range(epochs):
    dynamic_net.train()
    prev_state_epoch = {k: v.detach().cpu().clone() for k, v in dynamic_net.state_dict().items()}

    # Shuffle queries
    perm = torch.randperm(N, device=device)

    epoch_train_loss = 0.0
    num_batches = int(np.ceil(N / batch_size))
    for b in range(num_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, N)
        if s >= e: continue

        # --- queries in this minibatch ---
        q_idx  = perm[s:e]                               # (B,)
        yq_b   = y_train_t.index_select(0, q_idx)        # (B,)

        # --- uniformly sampled partners for these queries ---
        i_idx  = sample_uniform_partners(q_idx, K_SAMPLES, N)     # (B,K)

        # gather teacher params for partners
        mu_b    = mu_teacher_t.index_select(0, i_idx.view(-1)).view(i_idx.shape)      # (B,K)
        sigma_b = sigma_teacher_t.index_select(0, i_idx.view(-1)).view(i_idx.shape)   # (B,K)

        # distances (scaled)
        d_b = rowwise_dists(locs_train_t, q_idx, i_idx) / (dist_scale + 1e-8)         # (B,K)

        # forward → weights
        W_b = dynamic_net(d_b.reshape(-1, 1)).reshape(d_b.shape)                      # (B,K)

        # loss & step
        loss = query_weighted_nll_loss(W_b, mu_b, sigma_b, yq_b)
        dyn_optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamic_net.parameters(), max_norm=1.0)
        dyn_optim.step()

        epoch_train_loss += loss.item()

        if not np.isfinite(loss.item()):
            print(f"\n!!! Training stopped at epoch {epoch+1}, batch {b+1} due to NaN/inf loss. !!!")
            break
    if not np.isfinite(loss.item()):
        break

    avg_train_loss = epoch_train_loss / num_batches

    # ---- ESS check (post-epoch) on a fresh uniform sample ----
    dynamic_net.eval()
    with torch.no_grad():
        M = int(min(ESS_SAMPLE_SIZE, N))
        sample_q = torch.randperm(N, device=device)[:M]                # (M,)
        sample_i = sample_uniform_partners(sample_q, K_SAMPLES, N)     # (M,K)
        d_s      = rowwise_dists(locs_train_t, sample_q, sample_i) / (dist_scale + 1e-8)
        W_s      = dynamic_net(d_s.reshape(-1, 1)).reshape(M, K_SAMPLES)
        mean_ess = float(torch.mean(ess_rows_from_raw(W_s)).item())

    if mean_ess < ESS_MIN_THRESHOLD:
        print(f"\n!!! Early stopping: mean ESS {mean_ess:.1f} < threshold {ESS_MIN_THRESHOLD:.1f}. "
              f"Reverting to previous epoch state. !!!")
        dynamic_net.load_state_dict(prev_state_epoch)  # revert
        best_dyn_state = prev_state_epoch
        break

    print(f"Dynamic Model | Epoch {epoch+1:05d} | NLL: {avg_train_loss:.6f} | mean ESS ≈ {mean_ess:.1f}")

    # Track best by training objective (optional)
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        best_dyn_state = {k: v.detach().cpu().clone() for k, v in dynamic_net.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Training stopped early at epoch {epoch+1}.")
        break

# restore best
dynamic_net.load_state_dict(best_dyn_state)
dynamic_net.eval()


# ==============================================================================
# SECTION 5.2 — POST-TRAINING SHARPNESS / SHAPE CONTROLS  [PyTorch]
# ==============================================================================
@torch.inference_mode()
def power_sharpness_rowsum(w_mat: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Row-sum–preserving power transform to sharpen/smooth without changing
    each row's total mass. alpha>1 → sharper (ESS↓); alpha<1 → smoother (ESS↑).
    """
    w_pow = torch.pow(w_mat + EPS, alpha)
    sum_orig = torch.sum(w_mat, dim=1, keepdim=True) + EPS
    sum_new  = torch.sum(w_pow, dim=1, keepdim=True) + EPS
    return w_pow * (sum_orig / sum_new)


# ==============================================================================
# SECTION 7 — FULL-FIELD VIZ: weights for ALL points per frame (no subsampling)
# ==============================================================================

print("\n--- Generating Full-Field Animation of the Learned Dynamic Kernel ---")
from matplotlib.animation import FuncAnimation

fig_anim, ax_anim = plt.subplots(figsize=(12, 7))

# Frames: sweep across x (same selection as before)
num_frames = 50
sorted_indices_anim = np.argsort(X_train[:, 0])  # NumPy for ordering the scatter
sampler_indices = np.linspace(0, len(sorted_indices_anim) - 1, num_frames, dtype=int)
indices_to_animate = sorted_indices_anim[sampler_indices]

# Choose post-training sharpness (row-sum preserving); 1.0 = as trained
alpha = 1.0

@torch.inference_mode()
def weights_full_field_for_query(qidx: int) -> torch.Tensor:
    """
    Return raw weights w(d(q, ·)) for ALL training points (N,), with the query's
    self-weight zeroed (just for visualization so it doesn't dominate).
    The distances are scaled by the global 'dist_scale'.
    """
    q = torch.tensor([qidx], device=device)
    qloc = locs_train_t.index_select(0, q)              # (1, D)
    d_all = torch.linalg.norm(locs_train_t - qloc, dim=-1) / (dist_scale + 1e-8)  # (N,)
    w_all = dynamic_net(d_all.unsqueeze(1)).squeeze(1)  # (N,)
    w_all[qidx] = 0.0                                   # drop self for viz
    # optional: apply sharpness with row-sum preservation (treat this as one row)
    w_all = power_sharpness_rowsum(w_all.unsqueeze(0), alpha=alpha).squeeze(0)
    return w_all

# Estimate a stable color scale (vmax) from a small probe of queries
with torch.inference_mode():
    N = locs_train_t.shape[0]
    Mprobe = min(50, N)                           # small probe to avoid O(N^2) pass
    probe_idx = torch.randperm(N, device=device)[:Mprobe].tolist()
    probe_ws = []
    for qidx in probe_idx:
        probe_ws.append(weights_full_field_for_query(int(qidx)))
    probe_stack = torch.stack(probe_ws, dim=0)    # (Mprobe, N)
    vmax = float(torch.quantile(probe_stack.flatten(), 0.99).item())
    del probe_ws, probe_stack  # free memory

num_train_points = X_train.shape[0]
scat = ax_anim.scatter(
    X_train[:, 0], y_train, c=np.zeros(num_train_points), s=35,
    alpha=0.8, cmap='viridis', vmin=0, vmax=vmax
)
fig_anim.colorbar(scat, ax=ax_anim, label='Dynamic Weight w(d)')

query_highlight = ax_anim.scatter([], [], c='red', s=400, marker='*',
                                  edgecolor='black', linewidth=1.5, zorder=5, label='Query Point')
vline = ax_anim.axvline(x=X_train[0, 0], color='red', linestyle='--', lw=1.5, alpha=0.8)

ax_anim.set_xlabel('Feature X (Scaled)')
ax_anim.set_ylabel('Target y')
ax_anim.legend()
ax_anim.grid(True, alpha=0.4)
title = ax_anim.set_title('Learned Dynamic Kernel — Full Field', fontsize=16)

def update_animation(frame_index):
    qidx = int(indices_to_animate[frame_index])

    # Full-field weights for ALL N points against the current query
    w_full = weights_full_field_for_query(qidx)                      # (N,) torch
    w_np = w_full.detach().cpu().numpy()

    # Update scatter colors/sizes
    scat.set_array(w_np)
    scat.set_sizes(w_np * 200 + 15)

    # Full-field ESS (from raw weights)
    s = w_np.sum() + 1e-8
    ess_q = float((s * s) / (np.sum(w_np * w_np) + 1e-8))

    # Move markers & update title
    query_x, query_y = X_train[qidx, 0], y_train[qidx]
    query_highlight.set_offsets([query_x, query_y])
    vline.set_xdata([query_x])

    title.set_text(
        f'Full-Field Kernel (Query {frame_index+1}/{num_frames})  |  ESS≈{ess_q:.1f}'
    )

    print(f"Processing full-field frame {frame_index+1}/{num_frames}...")
    return scat, query_highlight, vline, title

anim = FuncAnimation(fig_anim, update_animation, frames=len(indices_to_animate),
                     interval=200, blit=False)
output_filename = r"c:\github\EVT_Univariate\EVT_Classes\NN\student_weight_animation_fullfield.gif"
anim.save(output_filename, writer='pillow', fps=5)
plt.close(fig_anim)

print(f"\n✅ Full-field animation complete! Saved as '{output_filename}'")