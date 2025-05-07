import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    # Class variables for shared dataset
    cached_images = None
    cached_labels = None
    batch_size = 128
    dataset_device = None

    @classmethod
    def preload_dataset(cls, data_loader, sample_size=None):
        """
        Preload the entire dataset into memory once for all model instances
        
        Args:
            data_loader: PyTorch DataLoader with the dataset
            sample_size: Optional int or float. If int, number of samples to keep.
                        If float between 0 and 1, fraction of dataset to keep.
                        If None, keep all data.
        """
        if cls.cached_images is not None and cls.cached_labels is not None:
            print("Dataset already preloaded")
            return
        if cls.dataset_device is None:
            raise ValueError("Dataset device not set. Set it before preloading.")

        print("Preloading dataset...")
        start = time.time()

        all_images = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                all_images.append(images)
                all_labels.append(labels)


        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)
        

        if sample_size is not None:
            original_size = len(all_labels)
            

            if isinstance(sample_size, float) and 0 < sample_size < 1:
                sample_size = int(original_size * sample_size)
            

            if not isinstance(sample_size, int) or sample_size <= 0:
                raise ValueError("sample_size must be positive int or float between 0 and 1")
                

            sample_size = min(sample_size, original_size)
            

            unique_labels = torch.unique(all_labels)
            num_classes = len(unique_labels)
            samples_per_class = sample_size // num_classes
            remaining_samples = sample_size % num_classes
            
            sampled_indices = []
            

            for label in unique_labels:
                class_indices = torch.where(all_labels == label)[0]
                

                extra = 1 if remaining_samples > 0 else 0
                if extra:
                    remaining_samples -= 1
                    

                if len(class_indices) <= samples_per_class + extra:

                    selected = class_indices
                else:

                    perm = torch.randperm(len(class_indices))
                    selected = class_indices[perm[:samples_per_class + extra]]
                
                sampled_indices.append(selected)
            

            sampled_indices = torch.cat(sampled_indices)
            

            all_images = all_images[sampled_indices]
            all_labels = all_labels[sampled_indices]
            
            print(f"Reduced dataset from {original_size} to {len(all_labels)} samples")
            

            new_dist = torch.bincount(all_labels)
            print(f"Class distribution: {new_dist.tolist()}")
        

        cls.cached_images = all_images.to(cls.dataset_device)
        cls.cached_labels = all_labels.to(cls.dataset_device)

        print(
            f"Dataset preloaded: {len(cls.cached_labels)} samples in {time.time() - start:.2f} seconds"
        )
        return cls.cached_images, cls.cached_labels
    def __init__(self, small=False):
        super(CNN, self).__init__()
        if not small:
            # convolutional layers
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            )

            # batch norm
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

            # maxpooling
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # fully conected layers
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
        else:
            # Convolutional layers
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, padding=1
            )
            self.conv2 = nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            )

            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # After 2 pooling layers on 8x8 → 4x4 → 2x2, so feature map is 32 x 2 x 2
            self.fc1 = nn.Linear(32 * 2 * 2, 64)
            self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def evaluate_cached(self, verbose=False):
        """Evaluate the model using the class-level preloaded dataset"""
        if CNN.cached_images is None or CNN.cached_labels is None:
            raise ValueError("Dataset not preloaded. Call CNN.preload_dataset() first.")

        if verbose:
            print("Evaluating model (using cached data)...")
            start = time.time()

        # Move model to the same device as the dataset
        self.to(CNN.dataset_device)

        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Process in batches to avoid memory issues with larger datasets
            for i in range(0, len(CNN.cached_labels), CNN.batch_size):
                batch_images = CNN.cached_images[i : i + CNN.batch_size]
                batch_labels = CNN.cached_labels[i : i + CNN.batch_size]

                outputs = self(batch_images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

        if verbose:
            print(f"Fast evaluation time: {time.time() - start:.2f} seconds")

        return 100 * correct / total

    def final_evaluate(self, test_loader, verbose=False):
        """
        Evaluate the model using a data loader (without caching)
        Returns the percentage of correct predictions
        """
        if verbose:
            print("Evaluating model on test set...")
            start = time.time()

        # Get current device of the model
        device = next(self.parameters()).device

        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                # Move batch data to the same device as the model
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)

                # Calculate accuracy
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total

        if verbose:
            print(f"Test accuracy: {accuracy:.2f}%")
            if verbose:
                print(f"Evaluation time: {time.time() - start:.2f} seconds")

        return accuracy

    def train_adam(self, lr=1e-3, epochs=10, verbose=False):
        """
        Training the model using Adam optimizer and cached dataset
        """
        if CNN.cached_images is None or CNN.cached_labels is None:
            raise ValueError("Dataset not preloaded. Call CNN.preload_dataset() first.")

        if verbose:
            print("Training with cached dataset...")
            start = time.time()

        # Move model to the device where the dataset is stored
        self.to(CNN.dataset_device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        dataset_size = len(CNN.cached_labels)
        num_batches = (dataset_size + CNN.batch_size - 1) // CNN.batch_size

        self.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0

            # Shuffle indices for this epoch (important for training)
            indices = torch.randperm(dataset_size, device=CNN.dataset_device)

            for i in range(0, dataset_size, CNN.batch_size):
                # Get batch indices for current iteration
                batch_indices = indices[i : i + CNN.batch_size]

                # Get batch data using indexed selection
                batch_images = CNN.cached_images[batch_indices]
                batch_labels = CNN.cached_labels[batch_indices]

                optimizer.zero_grad()
                outputs = self(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / num_batches
            if verbose and (epochs <= 10 or epoch % (epochs // 10) == 0):
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        if verbose:
            print(f"Training completed in {time.time() - start:.2f} seconds")

        return self
