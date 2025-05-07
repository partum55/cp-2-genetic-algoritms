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
    training_batch_size = None
    eval_images = None
    eval_labels = None
    @classmethod
    def preload_dataset(cls, data_loader, sample_size=None, variation_factor=0.05):
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
            base_samples_per_class = sample_size // num_classes
            
    
            variations = torch.FloatTensor(num_classes).uniform_(1-variation_factor, 1+variation_factor)

            variations = variations * (num_classes / variations.sum())
            

            samples_per_class_varied = [int(base_samples_per_class * v) for v in variations]
            

            total_selected = sum(samples_per_class_varied)
            difference = sample_size - total_selected
            

            indices_to_adjust = torch.randperm(num_classes)[:abs(difference)]
            for idx in indices_to_adjust:
                samples_per_class_varied[idx] += 1 if difference > 0 else -1
                

            min_samples = max(1, base_samples_per_class // 4)
            for i in range(len(samples_per_class_varied)):
                if samples_per_class_varied[i] < min_samples:
                    samples_per_class_varied[i] = min_samples
            
            sampled_indices = []
            for i,label in enumerate(unique_labels):
                class_indices = torch.where(all_labels == label)[0]
                target_samples = samples_per_class_varied[i]

                if len(class_indices) <= target_samples:

                    selected = class_indices
                else:

                    perm = torch.randperm(len(class_indices))
                    selected = class_indices[perm[:target_samples]]
                
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

    @classmethod
    def prepare_evaluation_batch(cls, sample_size=None, variation_factor=0.1, seed=None):
        """
        Prepare a consistent evaluation batch to be used for all models in current generation
        
        Args:
            sample_size: Number of samples for evaluation batch
            variation_factor: How much to vary from perfect balance
            seed: Random seed for reproducibility
        """
        if cls.cached_images is None or cls.cached_labels is None:
            raise ValueError("Dataset not preloaded. Call CNN.preload_dataset() first.")
        # print(f"Preparing evaluation batch with sample size: {sample_size}")
        # Set seed if provided for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            
        if sample_size is None or sample_size >= len(cls.cached_labels):
            cls.eval_images = cls.cached_images
            cls.eval_labels = cls.cached_labels
            return

        all_labels = cls.cached_labels
        unique_labels = torch.unique(all_labels)
        num_classes = len(unique_labels)

        samples_per_class = sample_size // num_classes

        variations = torch.FloatTensor(num_classes).uniform_(1-variation_factor, 1+variation_factor)
        variations = variations * (num_classes / variations.sum())  # Normalize to maintain total
        
        samples_per_class_varied = [max(1, int(samples_per_class * v)) for v in variations]

        total_samples = sum(samples_per_class_varied)
        if total_samples != sample_size:

            diff = sample_size - total_samples
            indices = torch.randperm(num_classes)[:abs(diff)]
            for idx in indices:
                samples_per_class_varied[idx] += 1 if diff > 0 else -1

        selected_indices = []
        for i, label in enumerate(unique_labels):

            class_indices = torch.where(all_labels == label)[0]

            if len(class_indices) <= samples_per_class_varied[i]:
                selected = class_indices
            else:
                perm = torch.randperm(len(class_indices))
                selected = class_indices[perm[:samples_per_class_varied[i]]]
            selected_indices.append(selected)

        selected_indices = torch.cat(selected_indices)
        perm = torch.randperm(len(selected_indices))
        selected_indices = selected_indices[perm]

        cls.eval_images = cls.cached_images[selected_indices]
        cls.eval_labels = cls.cached_labels[selected_indices]

        # new_dist = torch.bincount(cls.eval_labels)
        # print(f"Class distribution: {new_dist.tolist()}")
        if seed is not None:
            torch.manual_seed(torch.initial_seed())

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

    def evaluate_cached(self, verbose=False, use_prepared_batch=True):
        """
        Evaluate using the prepared evaluation batch for consistency within a generation
        """
        if CNN.cached_images is None or CNN.cached_labels is None:
            raise ValueError("Dataset not preloaded. Call CNN.preload_dataset() first.")

        if verbose:
            print("Evaluating model (using cached data)...")
            start = time.time()

        self.to(CNN.dataset_device)

        if use_prepared_batch and CNN.eval_images is not None and CNN.eval_labels is not None:
            eval_images = CNN.eval_images
            eval_labels = CNN.eval_labels
        else:
            # Fall back to using full dataset
            eval_images = CNN.cached_images
            eval_labels = CNN.cached_labels

        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Process in batches to avoid memory issues
            for i in range(0, len(eval_labels), CNN.batch_size):
                batch_images = eval_images[i : i + CNN.batch_size]
                batch_labels = eval_labels[i : i + CNN.batch_size]

                outputs = self(batch_images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)

        accuracy = 100 * correct / total

        if verbose:
            data_desc = f"sampled {total}" if sample_size else "full dataset"
            print(f"Accuracy on {data_desc}: {accuracy:.2f}%")
            print(f"Evaluation time: {time.time() - start:.2f} seconds")

        return accuracy

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
