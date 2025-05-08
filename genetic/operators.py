import copy
import random

import numpy as np
import torch

from model.cnn import CNN


def fitness_function(model, data_loader):
    return model.evaluate(data_loader)



def selection_tournament(population, fitnesses, num=2, tournament_size=3, selection_probability=0.75):
    """
    Tournament selection.

    Parameters:
    - population: list of individuals
    - fitnesses: list of fitness values
    - num: number of individuals to select (>= 2)
    - tournament_size: number of contestants in each tournament
    - selection_probability: probability that the best individual in the tournament wins
    Returns:
    - list of selected individuals (winners)

    Notes:
    - Higher tournament_size values create stronger selection pressure
    - For tournament_size=1, selection becomes completely random
    - The same individual may be selected multiple times
    """
    selected = []
    pop_size = len(population)

    for _ in range(num):
        # Choose random individuals for tournament
        indices = random.sample(range(pop_size), tournament_size)
        tournament = [(i, fitnesses[i]) for i in indices]
        tournament.sort(key=lambda x: x[1], reverse=True)  # sort by fitness

        # Probabilistically pick winner from sorted tournament
        for rank, (idx, _) in enumerate(tournament):
            if random.random() < selection_probability * ((1 - selection_probability) ** rank):
                selected.append(population[idx])
                break
        else:
            selected.append(population[tournament[-1][0]])

    return selected


def selection_roulette(population, fitnesses, num=2):
    """
    Roulette wheel selection

    Parameters:
    - population: list of individuals
    - fitnesses: list of fitness values
    - num: num of individuals to select

    Returns:
    - list of selected individuals

    Notes:
    - If all fitness values are zero, selection becomes completely random.
    - For small populations or when fitness differences are minimal, consider using
      rank-based or tournament selection instead to maintain selection pressure.
    """
    total = sum(fitnesses)

    # in case all fitnesses are 0
    if total == 0:
        return random.choice(population, k=num)

    probabilities = [f / total for f in fitnesses]

    return random.choices(population, weights=probabilities, k=num)


def selection_boltzmann(
    population,
    fitnesses,
    generation,
    num=2,
    initial_temp=100.0,
    cooling_rate=0.95,
    min_temp=0.1,
):
    """
    Boltzmann selection with exponential cooling

    Parameters:
    - population: list of individuals
    - fitnesses: list of fitness values
    - num: number of individuals to select
    - generation: current generation number
    - initial_temp: starting temperature
    - cooling_rate: how fast temperature decreases
    - min_temp: minimum temperature threshold

    Returns:
    - list of selected individuals

    Notes:
    - Early generations (high temperature): More exploration, diverse selection
    - Late generations (low temperature): More exploitation, focused on best solutions
    - Same individual may be selected multiple times
    - For temperature → ∞, selection becomes completely random
    - For temperature → 0, selection becomes deterministic (only best individual)
    """
    # exponential cooling
    current_temp = max(initial_temp * (cooling_rate**generation), min_temp)

    tf = torch.tensor(fitnesses, dtype=torch.float32)
    boltzmann = torch.exp(tf / current_temp)
    probabilities = boltzmann / boltzmann.sum()

    selected = torch.multinomial(probabilities, num, replacement=True)
    return [population[i] for i in selected]


def selection_rank_linear(population, fitnesses, num=2, selection_pressure=1.5):
    """
    Rank selection with linear ranging.

    Parameters:
    - population: list of individuals
    - fitnesses: list of fitness
    - num: num of individuals to select
    - selection_pressure: selection pressure (the higher the pressure the more disperse ranks) (1.0 < pressure ≤ 2.0)

    Returns:
    - list of selected individuals

    Notes:
    - Works well when fitness values are very close together
    - Less susceptible to dominance by few super-individuals
    - Same individual may be selected multiple times

    """
    sorted_pop = [ind for _, ind in sorted(zip(fitnesses, population), reverse=True)]

    n = len(population)
    ranks = torch.linspace(
        selection_pressure, 2 - selection_pressure, steps=n, dtype=torch.float32
    )

    probabilities = ranks / ranks.sum()

    selected = torch.multinomial(probabilities, num, replacement=True)
    return [sorted_pop[i] for i in selected]


def selection_rank_exponential(population, fitnesses, num=2, selection_pressure=1.5):
    """
    Rank selection with exponential ranging.

    Parameters:
    - population: list of individuals
    - fitnesses: list of fitness
    - num: num of individuals to select
    - selection_pressure: selection pressure (the higher the pressure the more disperse ranks) (1.0 < pressure ≤ 2.0)

    Returns:
    - list of selected individuals

    Notes:
    - Provides stronger bias toward top-ranked individuals than linear version
    - Effective in later generations when convergence is needed
    - Same individual may be selected multiple times
    """
    sorted_pop = [
        ind
        for _, ind in sorted(
            zip(fitnesses, population), key=lambda x: x[0], reverse=True
        )
    ]

    ranks = torch.arange(len(population), dtype=torch.float32)

    probabilities = torch.exp(-ranks / selection_pressure)
    probabilities /= probabilities.sum()

    selected_indices = torch.multinomial(probabilities, num, replacement=True)
    return [sorted_pop[i] for i in selected_indices]


def crossover(parent1, parent2, small):
    child = CNN(small).to(CNN.dataset_device)
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        if torch.rand(1).item() > 0.5:
            child_param.data.copy_(param1.data)
        else:
            child_param.data.copy_(param2.data)
    return child


def crossover_blend(parent1, parent2, small, alpha=0.5):
    child = CNN(small).to(CNN.dataset_device)
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        child_param.data.copy_(alpha * param1.data + (1 - alpha) * param2.data)
    return child


def crossover_mask(parent1, parent2, small):
    child = CNN(small).to(CNN.dataset_device)
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        mask = torch.rand_like(param1) > 0.5
        child_param.data.copy_(torch.where(mask, param1.data, param2.data))
    return child


def crossover_two_point(parent1, parent2, small):
    child = CNN(small).to(CNN.dataset_device)
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        # Flatten parameters
        flat1 = param1.data.view(-1)
        flat2 = param2.data.view(-1)
        length = flat1.size(0)

        if length < 2:
            # Not enough data for two-point crossover, just pick one
            child_param.data.copy_(
                param1.data if torch.rand(1).item() > 0.5 else param2.data
            )
            continue

        # Select two crossover points
        point1, point2 = sorted(random.sample(range(length), 2))

        # Create child flat tensor
        child_flat = torch.empty_like(flat1)
        child_flat[:point1] = flat1[:point1]
        child_flat[point1:point2] = flat2[point1:point2]
        child_flat[point2:] = flat1[point2:]

        # Reshape and copy to child
        child_param.data.copy_(child_flat.view_as(param1.data))

    return child


def crossover_one_point(parent1, parent2, small):
    child = CNN(small).to(CNN.dataset_device)
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        # Flatten parameters
        flat1 = param1.data.view(-1)
        flat2 = param2.data.view(-1)
        length = flat1.size(0)

        if length < 2:
            # Not enough data for one-point crossover, just pick one
            child_param.data.copy_(
                param1.data if torch.rand(1).item() > 0.5 else param2.data
            )
            continue

        # Select crossover point
        point = random.randint(0, length - 1)

        # Create child flat tensor
        child_flat = torch.empty_like(flat1)
        child_flat[:point] = flat1[:point]
        child_flat[point:] = flat2[point:]

        # Reshape and copy to child
        child_param.data.copy_(child_flat.view_as(param1.data))

    return child


def mutate(model, mutation_rate=0.1, scale=0.15):
    for name, param in model.named_parameters():
        if "weight" in name:
            if torch.rand(1) < mutation_rate:
                param.data += scale * torch.randn_like(param)
    return model

def improved_mutate(model, mutation_rate=0.1, scale=0.15, fitness_rank=None, total_ranks=None):
    """
    A more thoughtful mutation strategy for neural networks
    
    Args:
        model: The neural network model to mutate
        mutation_rate: Base mutation probability per parameter
        scale: Base scale for mutations
        fitness_rank: Rank of this model in the population (1 = best)
        total_ranks: Total number of ranks in the population
    """
    # Apply adaptive mutation if fitness information is available
    if fitness_rank is not None and total_ranks is not None:
        # Scale mutation rate and scale based on fitness rank
        # Lower fitness = higher mutation rate and scale
        fitness_factor = (fitness_rank / total_ranks) * 1.5  # 0.0-1.5 range
        mutation_rate = mutation_rate * (1 + fitness_factor)
        scale = scale * (1 + fitness_factor * 0.5)
    
    # Layer-specific factors
    layer_factors = {
        'conv1': 0.7,  # Early layers - more sensitive, mutate less
        'conv2': 0.9,
        'bn1': 0.5,    # Batch norm layers - very sensitive
        'bn2': 0.5,
        'fc1': 1.2,    # Hidden layers - mutate more
        'fc2': 1.5     # Output layer - mutate most
    }

    for name, param in model.named_parameters():
        # Extract layer name
        layer_name = name.split('.')[0]
        
        # Get layer-specific mutation factor
        factor = layer_factors.get(layer_name, 1.0)
        
        # Calculate effective mutation rate
        effective_rate = mutation_rate * factor
        
        # Different mutation for weights vs biases
        if 'weight' in name:
            if torch.rand(1) < effective_rate:
                # For weights, apply scaled noise
                param.data += scale * torch.randn_like(param)
        elif 'bias' in name:
            if torch.rand(1) < effective_rate * 0.7:  # Lower rate for biases
                # For biases, apply smaller mutations
                param.data += scale * 0.5 * torch.randn_like(param)
    
    return model