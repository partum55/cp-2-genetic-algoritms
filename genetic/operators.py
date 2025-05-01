import random
import copy
import torch

def fitness_function(model, data_loader):
    return model.evaluate(data_loader)

def selection_tournament(population, fitnesses, num=2, tournament_size=3, selection_probability=0.75):
    """
    Tournament selection.

    Parameters:
    - population: list of individuals
    - fitnesses: list of fitness value
    - num: num of individuals to select (>= 2)
    - tournament_size: num of contestants in the tournament
    - selection_probability: probability of the strongest individual to win
    
    Returns:
    - selected individuals (winners)

    Notes:
    - Higher tournament_size values create stronger selection pressure
    - For tournament_size=1, selection becomes completely random
    - The same individual may be selected multiple times
    """
    participants = list(range(len(population)))
    tournaments = [random.sample(participants, tournament_size) for _ in range(num)]

    winners = []
    for contestants in tournaments:
        if random.random() < selection_probability:
            winner_idx = max(contestants, key=lambda i: fitnesses[i])
        else:
            winner_idx = random.choice(contestants)
        winners.append(population[winner_idx])
    
    return winners

def selection_roulette(population, fitnesses, num=2):
    '''
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
    '''
    total = sum(fitnesses)
    
    # in case all fitnesses are 0
    if total == 0:
        return random.choice(population, k=num)
    
    probabilities = [f / total for f in fitnesses]

    return random.choice(population, weights=probabilities, k=num)

def selection_boltzmann(population, fitnesses, generation, num=2, initial_temp=100.0, cooling_rate=0.95, min_temp=0.1):
    '''
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
    '''
    # exponential cooling
    current_temp = max(initial_temp * (cooling_rate ** generation), min_temp)

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
    ranks = torch.linspace(selection_pressure, 2 - selection_pressure, steps=n, dtype=torch.float32)

    probabilities = ranks / ranks.sum()

    selected = torch.multinomial(probabilities, num, replacement=True)
    return [sorted_pop[i] for i in selected]

def selection_rank_exponential(population, fitnesses, num=2, selection_pressure=1.5):
    '''
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
    '''
    sorted_pop = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]

    ranks = torch.arange(len(population), dtype=torch.float32)

    probabilities = torch.exp(-ranks / selection_pressure)
    probabilities /= probabilities.sum()

    selected_indices = torch.multinomial(probabilities, num, replacement=True)
    return [sorted_pop[i] for i in selected_indices]

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
