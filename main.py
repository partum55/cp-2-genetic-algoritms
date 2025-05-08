import os


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def cellular_genetic_training(
    filename,
    synchronous,
    grid_size,
    neighborhood_type,
    selection_type,
    wrapped=True,
    small_mnist=False,
    epochs=100,
    batch_size=64,
    training_batch_size=500,
    save_model=True,
):
    """
    Run the Cellular Genetic Algorithm with the specified parameters.

    Parameters:
    - grid_size: Size of the grid population (e.g., 10 for a 10x10 grid).
    - neighborhood_type: Neighborhood structure for the cellular automata.
    - selection_type: Selection method to use (e.g., 'tournament', 'roulette').
    - wrapped: Whether to wrap around the edges of the grid.
    - small_mnist: Whether to use the small MNIST dataset.
    - epochs: Number of generations to run.

    Returns:
    - None
    """
    import csv

    import torch

    from model.cnn import CNN

    if small_mnist:
        from model.small_data_load import test_loader, train_loader
    else:
        from model.data_load import test_loader, train_loader
    from automata.fsm import AsyncCEA, SyncCEA

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting the program...")
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(f"Using device: {device}")

    print(f"Using small_mnist: {small_mnist}")
    print(f"Using grid size: {grid_size}")
    print(f"Using neighborhood type: {neighborhood_type}")
    print(f"Using selection type: {selection_type}")
    print(f"Using wrapped: {wrapped}")
    print(f"Using epochs: {epochs}")

    # Set up the CNN cashed dataset and device
    CNN.dataset_device = device
    CNN.batch_size = batch_size  # Set the batch size for the dataset
    ### for BIG MNIST dataset train_loader has 50000, san
    CNN.preload_dataset(train_loader, sample_size=0.04)
    CNN.prepare_evaluation_batch(
        sample_size=training_batch_size, variation_factor=0.05, seed=0
    )

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Generation", "Best fitness", "Worst fitness", "Average fitness"]
        )

        if synchronous:
            automat = SyncCEA(
                grid_size,
                neighborhood_type,
                selection_type,
                small_mnist=small_mnist,
                training_batch_size=training_batch_size,
                variation_factor=0.05,
            )
        else:
            automat = AsyncCEA(
                grid_size,
                neighborhood_type,
                selection_type,
                small_mnist=small_mnist,
                training_batch_size=training_batch_size,
                variation_factor=0.05,
            )

        # Run the algorithm for the specified number of epochs
        for _ in range(epochs):
            best, worst, average = automat.create_next_gen()
            writer.writerow([automat.gen, best, worst, average])
            if automat.gen % 5 == 0:
                print(f"Gen {automat.gen}: {best}/{worst}, avg {average}")
                f.flush()

        best_model = automat.get_final_model()
        print(
            f"Final test result on {automat.gen} is {best_model.final_evaluate(test_loader)}"
        )
    if save_model:
        model_name = "best_sync_cea" if synchronous else "best_async_cea"
        model_path = f"saved_models/{model_name}.pth"
        ensure_dir("saved_models")
        torch.save(best_model.state_dict(), "saved_models/cnn_model.pth")
        print("CNN model saved to saved_models/cnn_model.pth")
    return best_model


def adam_training(
    epochs=20,
    batch_size=64,
    small_mnist=False,
    save_model=True,
    model_name="adam_cnn_model.pth",
):
    """
    Train a CNN model using the Adam optimizer.

    Parameters:
    - epochs: Number of epochs to train the model.
    - batch_size: Batch size for training.
    - small_mnist: Whether to use the small MNIST dataset.

    Returns:
    - None
    """
    import torch

    from model.cnn import CNN

    if small_mnist:
        from model.small_data_load import test_loader, train_loader
    else:
        from model.data_load import test_loader, train_loader

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting Adam training...")
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(f"Using device: {device}")

    # Set up the CNN cashed dataset and device
    CNN.dataset_device = device
    CNN.batch_size = batch_size  # Set the batch size for the dataset
    CNN.preload_dataset(train_loader)

    # Initialize the model and train it using Adam optimizer
    model = CNN(device).to(device)
    model.train_adam(train_loader, epochs=epochs)
    if save_model:
        ensure_dir("saved_models")
        torch.save(model.state_dict(), f"saved_models/{model_name}")
        print(f"Model saved to saved_models/{model_name}")
    return model


if __name__ == "__main__":
    cellular_genetic_training(
        filename="Test_two_points_cross.csv",
        synchronous=False,
        grid_size=21,
        neighborhood_type=[
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 2, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ],
        selection_type="roulette",
        wrapped=True,
        small_mnist=False,
        epochs=1000,
        batch_size=64,
    )
