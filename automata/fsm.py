import heapq
from abc import ABC, abstractmethod

from genetic.operators import *
from model.cnn import CNN
import ast


class CellularEvolutionaryAutomata(ABC):
    def __init__(
        self,
        grid_size,
        neighborhood_type: list,
        selection_type,
        cross_over_method="two_point",
        training_batch_size=None,
        variation_factor=0.05,
        wrapped=True,
        small_mnist=False,
        mutation_type='gaussian',
    ):
        self.mutation_type = self.get_mutation_type(mutation_type)
        self.small_mnist = small_mnist
        self.training_batch_size = training_batch_size
        self.variation_rate = variation_factor
        self.cross_over_method = self.get_cross_over_method(cross_over_method)
        self.grid = self.create_grid_population(grid_size)
        self.width = grid_size
        self.height = grid_size
        self.fitness_table = self.create_fitness_hash_map()
        self.neighborhood_deltas = self.get_neighborhood_deltas(self.get_neighborhood_type(neighborhood_type))
        self.wrapped = wrapped
        self.selection_method = self.get_selection_method(selection_type)
        self.gen = 0
        # As of now one of ['rank_linear', 'tournament', 'roulette','rank_exponential']

    def create_grid_population(self, grid_size):
        if isinstance(grid_size, int) and grid_size >= 5:
            return [
                [CNN(self.small_mnist).to(CNN.dataset_device) for _ in range(grid_size)]
                for _ in range(grid_size)
            ]
        raise ValueError(
            "Grid size must be an integer greater than or equal to 5."
        )

    def get_mutation_type(self, mutation_type):
        if mutation_type == 'gaussian':
            return mutate
        if mutation_type == 'layer':
            return improved_mutate
        raise ValueError("No such mutation type.")
    def get_neighborhood_type(self, neighborhood_type):
        if neighborhood_type.lower() in ['m1', 'm2', 'c1', 'c2', 'fn1', 'fn2']:
            if neighborhood_type == "m1":
                return [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
            if neighborhood_type == "m2":
                return [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 2, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ]
            if neighborhood_type == "vn1":
                return [
                    [0, 1, 0],
                    [1, 2, 1],
                    [0, 1, 0],
                ]
            if neighborhood_type == "vn2":
                return [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 2, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]
                ]
            if neighborhood_type == "c1":
                return [
                    [0, 1, 0],
                    [1, 2, 1],
                    [0, 1, 0],
                ]
            if neighborhood_type == "c2":
                return [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 2, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0]
                ]
        else:
            try:
                neighborhood = ast.literal_eval(neighborhood_type)
            except:
                raise ValueError("Invalid --neighborhood format. Use Python list syntax.")

            if len(neighborhood) != len(neighborhood[0]):
                raise ValueError("Neighborhood type must be a square matrix.")
            if len(neighborhood) > self.width:
                raise ValueError("Neighborhood type is larger than grid size.")
            return neighborhood

        
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

    @staticmethod
    def get_cross_over_method(cross_over_method):
        if cross_over_method == "two_point":
            return crossover_two_point
        if cross_over_method == "blend":
            return crossover_blend
        if cross_over_method == "mask":
            return crossover_mask
        if cross_over_method == "simple":
            return crossover
        if cross_over_method == "one_point":
            return crossover_one_point
        raise ValueError("No such crossover method.")

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

    def create_fitness_hash_map(self):
        table = {}
        for y in range(self.height):
            for x in range(self.width):
                coord = (y, x)
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
        child = self.cross_over_method(parent_1, parent_2, self.small_mnist)
        child = self.mutation_type(child)
        return child

    @abstractmethod
    def create_next_gen(self): ...


class SyncCEA(CellularEvolutionaryAutomata):
    def create_next_gen(
        self,
        elite=0.1,
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


        for y in range(self.height):
            for x in range(self.width):
                coord = (y, x)
                if coord in top_k_coordinates:
                    new_grid[y][x] = self.grid[y][x]
                else:
                    child = self.get_child_from_cell(coord)
                    new_grid[y][x] = child

        self.grid = new_grid
        self.fitness_table = self.create_fitness_hash_map()
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

        for y in range(self.height):
            for x in range(self.width):
                coord = (y, x)
                self.fitness_table[coord] = self.grid[y][x].evaluate_cached()

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
