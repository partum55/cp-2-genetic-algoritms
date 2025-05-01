from abc import ABC, abstractmethod
from model.cnn import CNN
from genetic.operators import *
from model.data_load import test_loader, train_loader


class CellularEvolutionaryAutomata(ABC):
    def __init__(
        self,
        mutation_rate,
        grid_size,
        neighborhood_type: list,
        selection_type,
        wrapped=True,
    ):
        self.mutation_rate = mutation_rate
        self.grid = self.create_grid_population(grid_size)
        self.width = grid_size
        self.height = grid_size
        self.fitness_table = self.create_fitness_hash_map()
        assert all(isinstance(row, list) for row in neighborhood_type)
        self.neighborhood_deltas = self.get_neighborhood_deltas(neighborhood_type)
        self.wrapped = wrapped
        self.selection_method = self.get_selection_method(selection_type)
        self.gen = 0
        # As of now one of ['rank_linear', 'tournament', 'roulette','rank_exponential']

    def create_grid_population(self, grid_size):
        return [
            [CNN.create_random_model() for _ in range(grid_size)]
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
        if selection_type == 'rank_linear':
            return selection_rank_linear
        if selection_type == 'rank_exponential':
            return selection_rank_exponential
        if selection_type == 'tournament':
            return selection_tournament
        if selection_type == 'roulette':
            return selection_roulette
        raise ValueError('No such selection method.')

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

    def create_fitness_hash_map(self):
        table = {}
        for y in range(self.height):
            for x in range(self.width):
                table[(y, x)] = self.grid[y][x].evaluate(train_loader)
        return table

    def get_neighborhood_fitness(self, neighborhood_positions):
        fitness = []
        for pos in neighborhood_positions:
            fitness.append(self.fitness_table[pos])
        return fitness

    def get_best_train_fitness(self):
        return max(x for x in self.fitness_table.values())

    def get_final_fitness(self):
        return max([x.evaluate(test_loader) for row in self.grid for x in row])

    def get_child_from_cell(self, cell_pos):
        if self.wrapped:
            neighborhood_positions = self.get_neighborhood_wrapped(cell_pos)
        else:
            neighborhood_positions = self.get_neighborhood_bounded(cell_pos)
        neighborhood_fitness = self.get_neighborhood_fitness(neighborhood_positions)
        neighborhood = [self.grid[pos[0]][pos[1]] for pos in neighborhood_positions]
        parent_1, parent_2 = self.selection_method(neighborhood, neighborhood_fitness)
        child = crossover(parent_1, parent_2)
        child = mutate(child, self.mutation_rate)
        return child

    @abstractmethod
    def create_next_gen(self): ...


class SyncCEA(CellularEvolutionaryAutomata):
    def __init__(
        self, mutation_rate, grid_size, neighborhood_type, selection_type, wrapped=True
    ):
        super().__init__(
            mutation_rate, grid_size, neighborhood_type, selection_type, wrapped
        )

    def create_next_gen(self):
        new_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                child = self.get_child_from_cell((y,x))
                new_grid[y][x] = child
        self.grid = new_grid
        self.gen += 1
        self.fitness_table = self.create_fitness_hash_map()


class AsyncCEA(CellularEvolutionaryAutomata):
    def __init__(
        self, mutation_rate, grid_size, neighborhood_type, selection_type, wrapped=True
    ):
        super().__init__(
            mutation_rate, grid_size, neighborhood_type, selection_type, wrapped
        )

    def create_next_gen(self):
        for y in range(self.height):
            for x in range(self.width):
                child = self.get_child_from_cell((y,x))
                self.fitness_table[(y,x)] = child.evaluate(train_loader)
                self.grid[y][x] = child
        self.gen += 1
