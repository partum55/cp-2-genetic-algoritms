from automata.fsm import SyncCEA, AsyncCEA

neighborhood_type = \
    [[0,1,0],
     [1,2,1],
     [0,1,0]]
automat = SyncCEA(
            0.05,
            10,
            neighborhood_type,
            'rank_exponential'
                )
epochs = 20

for _ in range(epochs):
    print(f'Generation {automat.gen} best train fitness: {automat.get_best_train_fitness()}')
    automat.create_next_gen()
print(f'Generation {automat.gen} best train fitness: {automat.get_best_train_fitness()}')
print(f'Final test result on {automat.gen} is {automat.get_final_fitness()}')