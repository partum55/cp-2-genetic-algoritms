import heapq
from abc import ABC, abstractmethod

from genetic.operators import *
from model.cnn import CNN


class CellularEvolutionaryAutomata(ABC):
    def __init__(
        self,
        grid_size,
        neighborhood_type: list,
        selection_type,
        training_batch_size=None,
        variation_factor=0.05,
        wrapped=True,
        small_mnist=False,
    ):
        self.small_mnist = small_mnist
        self.training_batch_size = training_batch_size
        self.variation_rate = variation_factor
        self.grid = self.create_grid_population(grid_size)
        self.width = grid_size
        self.height = grid_size
        self.fitness_table = self.create_fitness_hash_map()
        self.neighborhood_deltas = self.get_neighborhood_deltas(neighborhood_type)
        self.wrapped = wrapped
        self.selection_method = self.get_selection_method(selection_type)
        self.gen = 0
        # As of now one of ['rank_linear', 'tournament', 'roulette','rank_exponential']

    def create_grid_population(self, grid_size):
        return [
            [CNN(self.small_mnist).to(CNN.dataset_device) for _ in range(grid_size)]
            for _ in range(grid_size)
        ]

    @staticmethod
    def get_neighborhood_deltas(neighborhood):
        positions = []
        center = None
        width = len(neighborhood)
        height = len(neighborhood[0])
        for j in range(height):
            for i in range(width):
                if neighborhood[j][i] == 1:
                    positions.append((j, i))
                elif neighborhood[j][i] == 2:
                    center = (j, i)
        return [(j[0] - center[0], j[1] - center[1]) for j in positions]

    @staticmethod
    def get_selection_method(selection_type):
        if selection_type == "rank_linear":
            return selection_rank_linear
        if selection_type == "rank_exponential":
            return selection_rank_exponential
        if selection_type == "tournament":
            return selection_tournament
        if selection_type == "roulette":
            return selection_roulette
        raise ValueError("No such selection method.")

    def get_neighborhood_wrapped(self, cell_pos):
        start_y, start_x = cell_pos
        positions = [(start_y, start_x)]
        for dy, dx in self.neighborhood_deltas:
            y = (start_y + dy) % self.height
            x = (start_x + dx) % self.width
            positions.append((y, x))
        return positions

    def get_neighborhood_bounded(self, cell_pos):
        start_y, start_x = cell_pos
        positions = [(start_y, start_x)]
        for dy, dx in self.neighborhood_deltas:
            y = start_y + dy
            x = start_x + dx
            if 0 <= x < self.width and 0 <= y < self.height:
                positions.append((y, x))
        return positions

    def get_neighborhood(self, cell_pos):
        return (
            self.get_neighborhood_wrapped(cell_pos)
            if self.wrapped
            else self.get_neighborhood_bounded(cell_pos)
        )

    def create_fitness_hash_map(self, reuse: dict = None):
        table = {}
        for y in range(self.height):
            for x in range(self.width):
                coord = (y, x)
                if reuse and coord in reuse:
                    table[coord] = reuse[coord]
                else:
                    table[coord] = self.grid[y][x].evaluate_cached()
        return table

    def get_neighborhood_fitness(self, neighborhood_positions):
        fitness = []
        for pos in neighborhood_positions:
            fitness.append(self.fitness_table[pos])
        return fitness

    def get_best_train_fitness(self):
        best_coord, best_val = max(self.fitness_table.items(), key=lambda x: x[1])
        return best_val

    def get_worst_train_fitness(self):
        worst_coord, worst_val = min(self.fitness_table.items(), key=lambda x: x[1])
        return worst_val

    def get_final_model(self):
        max_pair = max((pair[1], pair[0]) for pair in self.fitness_table.items())[1]
        y, x = max_pair
        return self.grid[y][x]

    def get_child_from_cell(self, cell_pos):
        neighborhood_positions = self.get_neighborhood(cell_pos)

        neighborhood_fitness = self.get_neighborhood_fitness(neighborhood_positions)
        neighborhood = [self.grid[pos[0]][pos[1]] for pos in neighborhood_positions]

        parent_1, parent_2 = self.selection_method(neighborhood, neighborhood_fitness)
        child = crossover_two_point(parent_1, parent_2, self.small_mnist)
        child = mutate(child)
        return child

    @abstractmethod
    def create_next_gen(self): ...


class SyncCEA(CellularEvolutionaryAutomata):
    def create_next_gen(
        self,
        elite=0.1,
        mutate_elite_fraction=0.5,
        elite_mutation_rate=0.04,
        elite_mutation_scale=1,
    ):
        CNN.prepare_evaluation_batch(
            sample_size=self.training_batch_size,
            variation_factor=self.variation_rate,
            seed=self.gen + 1,
        )
        new_grid = [[None] * self.width for _ in range(self.height)]
        elite_count = max(1, int(self.width * self.height * elite))
        top_k = sorted(self.fitness_table.items(), key=lambda x: x[1], reverse=True)[
            :elite_count
        ]
        top_k_coordinates = [x[0] for x in top_k]

        num_to_mutate = int(elite_count * mutate_elite_fraction)
        mutate_elite_coords = set(top_k_coordinates[-num_to_mutate:])

        for y in range(self.height):
            for x in range(self.width):
                coord = (y, x)
                if coord in top_k_coordinates:
                    model = self.grid[y][x]
                    if coord in mutate_elite_coords:
                        model = mutate(
                            model,
                            mutation_rate=elite_mutation_rate,
                            scale=elite_mutation_scale,
                        )
                    new_grid[y][x] = model
                else:
                    child = self.get_child_from_cell(coord)
                    new_grid[y][x] = child

        self.grid = new_grid
        elite_fitness = {
            coord: self.fitness_table[coord] for coord in top_k_coordinates
        }
        self.fitness_table = self.create_fitness_hash_map(reuse=elite_fitness)
        self.gen += 1
        return (
            self.get_best_train_fitness(),
            self.get_worst_train_fitness(),
            sum(self.fitness_table.values()) / len(self.fitness_table),
        )


class AsyncCEA(CellularEvolutionaryAutomata):

    def create_next_gen(self, elite=0.1):
        CNN.prepare_evaluation_batch(
            sample_size=self.training_batch_size,
            variation_factor=self.variation_rate,
            seed=self.gen + 1,
        )
        k = max(1, int(self.width * self.height * elite))

        top_k_heap = [(fitness, coord) for coord, fitness in self.fitness_table.items()]
        heapq.heapify(top_k_heap)
        top_k_heap = heapq.nlargest(k, top_k_heap)
        heapq.heapify(top_k_heap)

        top_k_coords_set = {coord for _, coord in top_k_heap}

        for y in range(self.height):
            for x in range(self.width):
                coord = (y, x)
                if coord in top_k_coords_set:
                    continue

                child = self.get_child_from_cell(coord)
                child_fitness = child.evaluate_cached()
                self.fitness_table[coord] = child_fitness
                self.grid[y][x] = child

                if child_fitness > top_k_heap[0][0]:
                    heapq.heappop(top_k_heap)
                    heapq.heappush(top_k_heap, (child_fitness, coord))
                    top_k_coords_set = {coord for _, coord in top_k_heap}
        self.gen += 1
        return (
            self.get_best_train_fitness(),
            self.get_worst_train_fitness(),
            sum(self.fitness_table.values()) / len(self.fitness_table),
        )
