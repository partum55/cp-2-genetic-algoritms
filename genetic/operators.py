import random
import copy
import torch

def fitness_function(model, data_loader):
    return model.evaluate(data_loader)

def selection_tornament_top_n(population, fitnesses, top_n=None):
    if top_n is None:
        top_n = len(population)
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    return [population[i] for i in sorted_indices[:top_n]]

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    flat_params1 = parent1.get_flat_params()
    flat_params2 = parent2.get_flat_params()
    crossover_point = random.randint(0, len(flat_params1))
    new_params = torch.cat((flat_params1[:crossover_point], flat_params2[crossover_point:]))
    child.set_flat_params(new_params)
    return child

def mutate(model, mutation_rate):
    mutated_model = copy.deepcopy(model)
    flat_params = mutated_model.get_flat_params()
    for i in range(len(flat_params)):
        if random.random() < mutation_rate:
            flat_params[i] += torch.randn(1).item()
    mutated_model.set_flat_params(flat_params)
    return mutated_model
