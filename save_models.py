import os
import torch
from model.cnn import CNN
from automata.fsm import SyncCEA, AsyncCEA
from model.data_load import train_loader, test_loader


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


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

    # Save the model
    ensure_dir("saved_models")
    torch.save(model.state_dict(), "saved_models/cnn_model.pth")
    print("CNN model saved to saved_models/cnn_model.pth")


def train_and_save_cea_models(grid_size=5, epochs=20):
    """Train CEA models and save the best individuals"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define neighborhood type
    # [0, 1, 0]
    # [1, 2, 1]
    # [0, 1, 0]
    neighborhood_type = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]

    # Train SyncCEA
    print("\nTraining SyncCEA model...")
    sync_cea = SyncCEA(grid_size, neighborhood_type, "rank_exponential", device)

    # Fix the mutation rate bug in SyncCEA by setting it explicitly
    sync_cea.mutation_rate = sync_cea.get_mutation_rate()

    for gen in range(epochs):
        print(f"Generation {sync_cea.gen} best train fitness: {sync_cea.get_best_train_fitness()}")
        sync_cea.create_next_gen()

    print(f"Final SyncCEA - Generation {sync_cea.gen} best train fitness: {sync_cea.get_best_train_fitness()}")

    # Get the best model from the grid
    best_fitness = -1
    best_model = None

    for y in range(sync_cea.height):
        for x in range(sync_cea.width):
            fitness = sync_cea.fitness_table[(y, x)]
            if fitness > best_fitness:
                best_fitness = fitness
                best_model = sync_cea.grid[y][x]

    # Save the best SyncCEA model
    ensure_dir("saved_models")
    torch.save(best_model.state_dict(), "saved_models/sync_cea_best.pth")
    print(f"Best SyncCEA model saved with fitness: {best_fitness:.2f}%")

    # Train AsyncCEA
    print("\nTraining AsyncCEA model...")
    async_cea = AsyncCEA(grid_size, neighborhood_type, "rank_exponential", device)

    # Fix the mutation rate bug in AsyncCEA by setting it explicitly
    async_cea.mutation_rate = async_cea.get_mutation_rate()

    for gen in range(epochs):
        print(f"Generation {async_cea.gen} best train fitness: {async_cea.get_best_train_fitness()}")
        async_cea.create_next_gen()

    print(f"Final AsyncCEA - Generation {async_cea.gen} best train fitness: {async_cea.get_best_train_fitness()}")

    # Get the best model from the grid
    best_fitness = -1
    best_model = None

    for y in range(async_cea.height):
        for x in range(async_cea.width):
            fitness = async_cea.fitness_table[(y, x)]
            if fitness > best_fitness:
                best_fitness = fitness
                best_model = async_cea.grid[y][x]

    # Save the best AsyncCEA model
    torch.save(best_model.state_dict(), "saved_models/async_cea_best.pth")
    print(f"Best AsyncCEA model saved with fitness: {best_fitness:.2f}%")


if __name__ == "__main__":
    # Train and save all models
    print("Starting training and saving models...")

    # Set a smaller number of epochs for faster execution
    num_epochs = 5
    grid_size = 3  # Smaller grid for quicker training

    # Train and save standard CNN model
    train_and_save_cnn_model(epochs=num_epochs)

    # Train and save CEA models
    train_and_save_cea_models(grid_size=grid_size, epochs=num_epochs)

    print("\nAll models trained and saved successfully!")
