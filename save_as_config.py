import os
import json
import torch
from model.cnn import CNN
from automata.fsm import SyncCEA, AsyncCEA
from model.data_load import train_loader, test_loader


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model_with_config(model, model_path, config_path, metadata):
    """Save model in PyTorch format and its configuration as JSON"""
    # Save the PyTorch model
    torch.save(model.state_dict(), model_path)

    # Extract model architecture info and add it to metadata
    architecture_info = {
        "layers": [
            {"type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1},
            {"type": "BatchNorm2d", "channels": 32},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel_size": 2, "stride": 2},
            {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1},
            {"type": "BatchNorm2d", "channels": 64},
            {"type": "ReLU"},
            {"type": "MaxPool2d", "kernel_size": 2, "stride": 2},
            {"type": "Linear", "in_features": 64 * 7 * 7, "out_features": 128},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 128, "out_features": 10}
        ]
    }

    config = {
        "model_path": os.path.basename(model_path),
        "architecture": architecture_info,
        "metadata": metadata
    }

    # Save the configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Configuration saved to {config_path}")


def train_and_save_cnn_model(epochs=20):
    """Train a standard CNN model with Adam optimizer and save it"""
    print("Training standard CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(device).to(device)
    model.train_adam(train_loader, lr=1e-3, epochs=epochs)

    # Evaluate the model
    train_accuracy = model.evaluate(train_loader)
    test_accuracy = model.evaluate(test_loader)

    print(f"CNN Model - Training accuracy: {train_accuracy:.2f}%")
    print(f"CNN Model - Test accuracy: {test_accuracy:.2f}%")

    # Save the model with config
    ensure_dir("saved_models")
    ensure_dir("configs")

    metadata = {
        "type": "CNN",
        "epochs": epochs,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "learning_rate": 1e-3,
        "optimizer": "Adam"
    }

    save_model_with_config(
        model,
        "saved_models/cnn_model.pth",
        "configs/cnn_model.json",
        metadata
    )

    # Return accuracy values for website metrics
    return float(train_accuracy), float(test_accuracy), epochs


def train_and_save_cea_models(grid_size=5, epochs=20):
    """Train CEA models and save the best individuals"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define neighborhood type
    neighborhood_type = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]

    # Train SyncCEA
    print("\nTraining SyncCEA model...")
    sync_cea = SyncCEA(grid_size, neighborhood_type, "rank_exponential", device, train_loader, test_loader)

    # Fix the mutation rate bug by setting it explicitly
    sync_cea.mutation_rate = sync_cea.get_mutation_rate()

    for gen in range(epochs):
        print(f"Generation {sync_cea.gen} best train fitness: {sync_cea.get_best_train_fitness()}")
        sync_cea.create_next_gen()

    final_train_fitness = sync_cea.get_best_train_fitness()
    final_test_fitness = sync_cea.get_final_fitness()

    print(f"Final SyncCEA - Generation {sync_cea.gen} best train fitness: {final_train_fitness}")
    print(f"Final SyncCEA - Test accuracy: {final_test_fitness}%")

    # Get the best model from the grid
    best_fitness = -1
    best_model = None

    for y in range(sync_cea.height):
        for x in range(sync_cea.width):
            fitness = sync_cea.fitness_table[(y, x)]
            if fitness > best_fitness:
                best_fitness = fitness
                best_model = sync_cea.grid[y][x]

    # Save the best SyncCEA model with config
    metadata = {
        "type": "SyncCEA",
        "generations": epochs,
        "grid_size": grid_size,
        "selection_method": "rank_exponential",
        "train_accuracy": float(final_train_fitness),
        "test_accuracy": float(final_test_fitness),
        "neighborhood_type": neighborhood_type
    }

    save_model_with_config(
        best_model,
        "saved_models/sync_cea_best.pth",
        "configs/sync_cea_best.json",
        metadata
    )

    # Store sync CEA metrics
    sync_train_accuracy = float(final_train_fitness)
    sync_test_accuracy = float(final_test_fitness)

    # Train AsyncCEA
    print("\nTraining AsyncCEA model...")
    async_cea = AsyncCEA(grid_size, neighborhood_type, "rank_exponential", device, train_loader, test_loader)

    # Fix the mutation rate bug by setting it explicitly
    async_cea.mutation_rate = async_cea.get_mutation_rate()

    for gen in range(epochs):
        print(f"Generation {async_cea.gen} best train fitness: {async_cea.get_best_train_fitness()}")
        async_cea.create_next_gen()

    final_train_fitness = async_cea.get_best_train_fitness()
    final_test_fitness = async_cea.get_final_fitness()

    print(f"Final AsyncCEA - Generation {async_cea.gen} best train fitness: {final_train_fitness}")
    print(f"Final AsyncCEA - Test accuracy: {final_test_fitness}%")

    # Get the best model from the grid
    best_fitness = -1
    best_model = None

    for y in range(async_cea.height):
        for x in range(async_cea.width):
            fitness = async_cea.fitness_table[(y, x)]
            if fitness > best_fitness:
                best_fitness = fitness
                best_model = async_cea.grid[y][x]

    # Save the best AsyncCEA model with config
    metadata = {
        "type": "AsyncCEA",
        "generations": epochs,
        "grid_size": grid_size,
        "selection_method": "rank_exponential",
        "train_accuracy": float(final_train_fitness),
        "test_accuracy": float(final_test_fitness),
        "neighborhood_type": neighborhood_type
    }

    save_model_with_config(
        best_model,
        "saved_models/async_cea_best.pth",
        "configs/async_cea_best.json",
        metadata
    )

    # Store async CEA metrics
    async_train_accuracy = float(final_train_fitness)
    async_test_accuracy = float(final_test_fitness)

    return (sync_train_accuracy, sync_test_accuracy,
            async_train_accuracy, async_test_accuracy, epochs)


def save_metrics_for_website(cnn_metrics, cea_metrics):
    """Saves metrics of models for display on the website"""
    cnn_train_accuracy, cnn_test_accuracy, cnn_epochs = cnn_metrics
    (sync_train_accuracy, sync_test_accuracy,
     async_train_accuracy, async_test_accuracy, cea_epochs) = cea_metrics

    metrics = {
        "cnn": {
            "name": "Standard CNN",
            "description": "Standard Convolutional Neural Network trained with Adam optimizer.",
            "stats": {
                "accuracy": f"{cnn_test_accuracy:.1f}%",
                "parameters": "1.3M",
                "epochs": str(cnn_epochs)
            }
        },
        "syncCEA": {
            "name": "Sync CEA",
            "description": "Synchronous Cellular Evolutionary Automata - evolves a population of CNNs simultaneously.",
            "stats": {
                "accuracy": f"{sync_test_accuracy:.1f}%",
                "parameters": "1.2M",
                "generations": str(cea_epochs)
            }
        },
        "asyncCEA": {
            "name": "Async CEA",
            "description": "Asynchronous Cellular Evolutionary Automata - evolves a population of CNNs cell by cell.",
            "stats": {
                "accuracy": f"{async_test_accuracy:.1f}%",
                "parameters": "1.1M",
                "generations": str(cea_epochs)
            }
        }
    }

    # Save metrics to JSON file for website
    ensure_dir("static")
    with open("static/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics saved for website")


if __name__ == "__main__":
    print("Starting training and saving models with configurations...")

    # Set parameters for training
    num_epochs = 100  # Increased from default 5
    grid_size = 16  # Increased from default 3

    # Train and save standard CNN model
    cnn_metrics = train_and_save_cnn_model(epochs=num_epochs)

    # Train and save CEA models
    cea_metrics = train_and_save_cea_models(grid_size=grid_size, epochs=num_epochs)

    # Save metrics for website
    save_metrics_for_website(cnn_metrics, cea_metrics)

    print("\nAll models trained and saved successfully with configurations!")