import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm # For plotting the final distribution

# --- 1. Generate Synthetic Data (Same as before) ---
TRUE_MEAN = 170.0
TRUE_STD_DEV = 8.0
N_SAMPLES = 1000

np.random.seed(42) # for reproducibility
y_heights = np.random.normal(loc=TRUE_MEAN, scale=TRUE_STD_DEV, size=N_SAMPLES)
y_heights_np = y_heights.reshape(-1, 1) # Keep numpy version for plotting later if needed

X_dummy_np = np.ones((N_SAMPLES, 1))

print(f"Generated {N_SAMPLES} data points.")
print(f"True Mean: {TRUE_MEAN:.2f}, True Std Dev: {TRUE_STD_DEV:.2f}")
print(f"Sample Mean: {np.mean(y_heights_np):.2f}, Sample Std Dev: {np.std(y_heights_np):.2f}")

# Optional: Visualize the generated data (using numpy array)
plt.figure(figsize=(8, 4))
plt.hist(y_heights_np, bins=30, density=True, alpha=0.7, label='Generated Heights')
plt.title('Distribution of Generated Adult Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.show()

# --- Convert data to PyTorch Tensors ---
X_dummy = torch.tensor(X_dummy_np, dtype=torch.float32)
y_heights = torch.tensor(y_heights_np, dtype=torch.float32)


# --- 2. Define Gaussian NLL Loss (PyTorch version) ---
def gaussian_nll_loss_pytorch(y_true, y_pred):
    """ Calculates the Gaussian Negative Log-Likelihood loss using PyTorch.

    Args:
        y_true: Ground truth values (shape: [batch_size, 1])
        y_pred: Predicted distribution parameters (shape: [batch_size, 2]).
                Column 0: predicted mean (mu)
                Column 1: predicted log standard deviation (log_sigma)
    Returns:
        Loss tensor (scalar).
    """
    mu = y_pred[:, 0:1]        # Extract predicted mean
    log_sigma = y_pred[:, 1:2] # Extract predicted log sigma

    # Ensure sigma is positive and numerically stable
    epsilon = 1e-6
    sigma = torch.exp(log_sigma) + epsilon

    # Calculate NLL components
    mse = torch.square(y_true - mu) # Or (y_true - mu)**2
    variance = torch.square(sigma) # Or sigma**2

    # NLL = log(sigma) + mse / (2 * variance)
    # log_likelihood = -log(sigma) - mse / (2 * variance)
    log_likelihood = -torch.log(sigma) - (mse / (2. * variance))

    # Return the mean negative log likelihood over the batch
    return -torch.mean(log_likelihood)

# --- 3. Build the Neural Network (PyTorch Module) ---
class DistributionNN(nn.Module):
    def __init__(self, input_dim):
        super(DistributionNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()

        # Output layers for mu and log_sigma
        self.mu_layer = nn.Linear(8, 1)
        self.log_sigma_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x)
        # Concatenate the outputs along the feature dimension (dim=1)
        outputs = torch.cat((mu, log_sigma), dim=1)
        return outputs

input_dim = X_dummy.shape[1]
model = DistributionNN(input_dim)
print("\nModel Architecture:")
print(model)

# --- 4. Set up Training Loop ---
LEARNING_RATE = 0.005 # Might need tuning
EPOCHS = 150

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Store loss history
loss_history = []

print("\nTraining the model...")
for epoch in range(EPOCHS):
    # Set model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_dummy)

    # Calculate loss
    loss = gaussian_nll_loss_pytorch(y_heights, outputs)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Record loss
    loss_history.append(loss.item())

    # Print progress (optional)
    if (epoch + 1) % 25 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("Training complete.")

# Optional: Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title('Model Training Loss (Gaussian NLL - PyTorch)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)
plt.show()

# --- 5. Evaluate and Interpret Results ---
# Set model to evaluation mode
model.eval()

# Predict distribution parameters (disable gradient calculation)
with torch.no_grad():
    predicted_params_tensor = model(X_dummy)

# Convert tensor to numpy array for easier handling/plotting
predicted_params = predicted_params_tensor.numpy()

# Extract the learned parameters (average over all predictions)
predicted_mu = np.mean(predicted_params[:, 0])
predicted_log_sigma = np.mean(predicted_params[:, 1])
predicted_sigma = np.exp(predicted_log_sigma)

print("\n--- Results ---")
print(f"True Distribution Parameters:       Mean = {TRUE_MEAN:.2f}, Std Dev = {TRUE_STD_DEV:.2f}")
print(f"Sample Distribution Parameters:     Mean = {np.mean(y_heights_np):.2f}, Std Dev = {np.std(y_heights_np):.2f}")
print(f"Predicted Distribution Parameters:  Mean = {predicted_mu:.2f}, Std Dev = {predicted_sigma:.2f}")


# Optional: Plot the learned distribution over the histogram (using numpy data)
plt.figure(figsize=(8, 4))
plt.hist(y_heights_np, bins=30, density=True, alpha=0.7, label='Generated Heights')
x_range = np.linspace(y_heights_np.min(), y_heights_np.max(), 200)
plt.plot(x_range, norm.pdf(x_range, TRUE_MEAN, TRUE_STD_DEV), 'r-', lw=2, label=f'True Dist ($\mu$={TRUE_MEAN:.1f}, $\sigma$={TRUE_STD_DEV:.1f})')
plt.plot(x_range, norm.pdf(x_range, predicted_mu, predicted_sigma), 'g--', lw=2, label=f'Predicted Dist ($\mu$={predicted_mu:.1f}, $\sigma$={predicted_sigma:.1f})')
plt.title('Generated Data and Fitted Gaussian Distribution (PyTorch)')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()