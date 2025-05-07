import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset, random_split

# Load the digits data (8x8 grayscale images, labels 0–9)
digits = load_digits()
X = digits.images  # shape: (1797, 8, 8)
y = digits.target

# Normalize and convert to PyTorch tensors
X_tensor = torch.tensor(X / 16.0, dtype=torch.float32).unsqueeze(1)  # (N, 1, 8, 8)
y_tensor = torch.tensor(y, dtype=torch.long)

# Wrap into TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split into train/test (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# # Example: iterate over one batch
# for images, labels in train_loader:
#     print("Batch images shape:", images.shape)
#     print("Batch labels shape:", labels.shape)
#     break
