
def main():
    print("Starting the program...")
    from automata.fsm import SyncCEA, AsyncCEA
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neighborhood_type = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
    automat = SyncCEA(10, neighborhood_type, "rank_exponential", device)
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

if __name__ == "__main__":
    main()