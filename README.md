# Artificial Life Simulation with Cellular Evolutionary Automata

## Project Description
This project implements a simulation of artificial life using Cellular Evolutionary Automata (CEA). The system models evolutionary processes in a grid-based environment, where neural networks evolve through genetic algorithms. The project is designed to explore emergent behaviors and optimization techniques inspired by natural evolution.

## Features
- **Neural Network Models**: Convolutional Neural Networks (CNNs) are used as individuals in the population.
- **Genetic Operators**: Includes selection, crossover, and mutation for evolving the population.
- **Cellular Automata**: Implements synchronous and asynchronous CEA for grid-based evolution.
- **Fitness Evaluation**: Evaluates individuals based on their performance on the MNIST dataset.

## Modules
### 1. `model/data_load.py`
Handles loading and preprocessing of the MNIST dataset. It defines `train_loader` and `test_loader` for training and testing the CNN models.

### 2. `model/cnn.py`
Defines the CNN architecture and provides utility methods for:
- Evaluating the model's accuracy.
- Getting and setting flattened parameters for genetic operations.
- Creating random models for initialization.

### 3. `genetic/operators.py`
Implements genetic operators:
- **Selection**: Includes tournament, roulette, Boltzmann, and rank-based selection methods.
- **Crossover**: Combines parameters from two parent models to create a child.
- **Mutation**: Introduces random changes to a model's parameters.

### 4. `automata/fsm.py`
Defines the Cellular Evolutionary Automata framework:
- **SyncCEA**: Updates the entire grid synchronously.
- **AsyncCEA**: Updates the grid asynchronously, cell by cell.
- Supports different neighborhood types and selection methods.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd cp-2-artificial-life
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Train and evolve the population:
   ```python
   ```

## Contributors


## License
This project is licensed under the MIT License.
