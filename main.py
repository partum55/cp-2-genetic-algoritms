
def main():
    print("Starting the program...")
    from automata.fsm import SyncCEA, AsyncCEA
    from model.data_load import train_loader, test_loader
    import torch
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neighborhood_type = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
    automat = AsyncCEA(10, neighborhood_type, "rank_exponential", device, train_loader, test_loader)
    epochs = 20

    for _ in range(epochs):
        print(
            f"Generation {automat.gen} best train fitness: {automat.get_best_train_fitness()}"
        )
        automat.create_next_gen()
    print(
        f"Generation {automat.gen} best train fitness: {automat.get_best_train_fitness()}"
    )
    print(f"Final test result on {automat.gen} is {automat.get_final_fitness()}")

def main_adam():
    import torch
    from model.data_load import train_loader, test_loader
    from model.cnn import CNN
    print("Starting Adam training...")
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(device).to(device)
    model.train_adam(train_loader, lr=1e-3, epochs=2)
    train_accuracy = model.evaluate(train_loader)
    test_accuracy = model.evaluate(test_loader)

    print(f"Training accuracy after 20 epochs: {train_accuracy:.2f}%")
    print(f"Test accuracy after 20 epochs: {test_accuracy:.2f}%")


if __name__ == "__main__":ї
    # main()
    # main_adam()             # comment one of the methods before running !!!