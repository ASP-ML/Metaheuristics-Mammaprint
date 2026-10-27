"""
Simulated Annealing (SA) for Feature Selection
==============================================

This module implements the Simulated Annealing metaheuristic algorithm
adapted for binary feature selection in the Mammaprint project.

Simulated Annealing is inspired by the annealing process in metallurgy,
where controlled cooling allows atoms to reach low-energy states.

Algorithm Overview:
------------------
1. Initialize with random feature subset and high temperature
2. For each iteration (move):
   a. Generate neighbor solution via perturbation (add or remove a feature)
   b. Evaluate fitness of neighbor
   c. Accept neighbor if better, or probabilistically if worse
   d. Reduce temperature according to cooling schedule
3. Return best solution found during search

Key Features:
------------
- Temperature-based acceptance probability
- Metropolis acceptance criterion for escaping local optima
- Geometric cooling after every move, from T₀ down to T_min
- Neighbourhood moves: add or remove one random feature
- Initial subset drawn from a Gaussian distribution (a feature is selected when its draw is > 0)
- JVM reload handling for Weka objective functions
- Minimum subset length enforcement

Mathematical Foundation:
-----------------------
Acceptance probability (maximization):
P(accept) = exp(-ΔE / T)

where:
- ΔE: fitness loss (current - neighbor) of a worse neighbour
- T: current temperature

Cooling schedule:
T(t + 1) = α · T(t), t = current move

The search stops when T falls below T_min (early stopping criterion) or
when the move budget is exhausted, whichever comes first.

References:
----------
Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by
simulated annealing. Science, 220(4598), 671-680.

Author: ASP-ML Team
License: MIT
Version: 1.1.0
"""

import math
import numpy as np
import logging
import weka.core.jvm as jvm
from .base import (Metaheuristic, writterHistoric, stream_key,
                   MAX_ITERATIONS_JVM_WEKA, SET_MIN_LENGTH, MIN_LENGTH_SUBSET)


class SimulatedAnnealing(Metaheuristic):
    """
    Simulated Annealing for binary feature selection optimization.

    Single-solution metaheuristic that explores the search space via
    controlled random perturbations with temperature-based acceptance.

    Attributes:
    ----------
    initTemp : float
        Initial temperature T₀ (higher = more exploration)
    min_temp : float
        Temperature T_min below which the search stops
    temperature : float
        Current temperature
    epochs : int
        Maximum number of iterations (moves)
    nFeatures : int
        Total number of features
    alpha : float
        Cooling rate (0 < α < 1)
    objectiveFunction : callable
        Fitness evaluation function
    bestSubset : np.ndarray
        Best feature subset found (local indexes)
    bestFitValue : float
        Fitness of best subset
    """
    def __init__(self, experiment_index, data, labels, initTemp, iterations, nFeatures, beta, alpha, objectiveFunction, type_of, indexes, path_historic, seed=None, min_temp=1e-4):
        """
        Initialize Simulated Annealing for feature selection.

        Parameters:
        ----------
        experiment_index : int
            Unique identifier for experimental run
        data : pd.DataFrame or np.ndarray
            Feature matrix without target variable
        labels : pd.Series or np.ndarray
            Target class labels
        initTemp : float
            Initial temperature T₀
        iterations : int
            Maximum number of moves
        nFeatures : int
            Total number of features available
        beta : float
            Unused, kept for backward compatibility of the constructor signature
        alpha : float
            Cooling rate applied after every move (higher = slower cooling)
        objectiveFunction : callable
            Function to evaluate feature subset quality
        type_of : str
            Objective function identifier
        indexes : array-like
            Feature indices mapping to original dataset
        path_historic : str
            CSV file path for logging optimization history
        seed : int or None
            Base seed of the private random generator (None = not reproducible)
        min_temp : float
            Temperature T_min below which the search stops (default 1e-4)
        """
        Metaheuristic.__init__(self)
        self.initTemp = initTemp
        self.min_temp = min_temp
        self.epochs = iterations
        self.objectiveFunction = objectiveFunction
        self.nFeatures = nFeatures
        self.alpha = alpha
        self.type_of = type_of
        self.name = 'SA'
        self.experiment_index = experiment_index
        self.path_historic = path_historic
        self.setDataSet(data, labels, indexes)
        self.set_seed(seed, stream_key(f'{self.name}_{self.type_of}'), nFeatures)
        self.initialize()

    def set_new_iterations(self, iterations):
        # Each restart budget gets its own reproducible random stream
        self.reseed(iterations)
        self.initialize()
        self.epochs = iterations
        self.bestSubset = []
        self.bestScoreValue = 0

    def setParams(self, params):
        self.reload_weka_jvm = params['reload_weka_jvm']
        self.current_generation = params['current_generation']
        self.currentSubset = params['currentSubset']
        self.currentFitValue = params['currentFitValue']
        self.bestSubset = params['bestSubset']
        self.bestFitValue = params['bestFitValue']
        self.temperature = params['temperature']
        self.n_evaluations = params['n_evaluations']
        self.rng.bit_generator.state = params['rng_state']

    def setExperimentIndex(self, experiment_index):
        self.experiment_index = experiment_index

    def getParams(self):
        info_dict = {
            'name': self.name,
            'of_name': self.type_of,
            'iterations': self.epochs,
            'settings': 'T0: ' + str(self.initTemp) + ' Tmin: ' + str(self.min_temp) + ' Alpha: ' + str(self.alpha) + ' N features: ' + str(self.nFeatures) + ' Evals: ' + str(self.n_evaluations),
            'best_subset': [self.original_indexes[i] for i in list(self.bestSubset)],
            'best_score_value': self.bestScoreValue,
            'experiment_index': self.experiment_index,
            'current_generation': self.current_generation,
            'reload_weka_jvm': self.reload_weka_jvm,
            'currentSubset': self.currentSubset,
            'currentFitValue': self.currentFitValue,
            'bestSubset': self.bestSubset,
            'bestFitValue': self.bestFitValue,
            'temperature': self.temperature,
            'n_evaluations': self.n_evaluations,
            'rng_state': self.rng.bit_generator.state
        }

        return info_dict

    def setDataSet(self, data, labels, indexes):
        self.train_data = data
        self.labels = labels
        self.original_indexes = indexes
        print(f'Set train dataset: {self.name} - {self.train_data.shape}')

    def initialize(self):
        self.historic = []
        self.nFeatures = self.train_data.shape[1]
        self.reload_weka_jvm = False
        self.current_generation = 0
        self.currentSubset = None
        self.currentFitValue = 0
        self.bestSubset = []
        self.bestFitValue = 0
        self.temperature = self.initTemp
        self.n_evaluations = 0

    def runProcess(self):
        jvm.start(packages=True, logging_level=logging.INFO)
        _ = self.run()
        jvm.stop()
        return self.getParams()

    def printF(self, msg):
        print(f'>> {self.name} - {self.type_of}: ', msg)

    def debug(self, selection):
        """Repair a boolean feature selection with fewer than MIN_LENGTH_SUBSET features."""
        self.printF('Executing debug')
        selection = np.array(selection, dtype=bool)
        min_length = min(MIN_LENGTH_SUBSET, self.nFeatures)
        while selection.sum() < min_length:
            selection[self.rng.integers(0, self.nFeatures)] = True
        return selection

    def acceptance_probability(self, current_fit, new_fit):
        """Metropolis criterion for a worse neighbour (maximization): P = exp(-ΔE / T)."""
        return math.exp(-(current_fit - new_fit) / self.temperature)

    def perturb(self, current):
        """Add or remove one feature; None if no feasible move exists."""
        remaining = np.setdiff1d(np.arange(self.nFeatures), current)
        moves = []
        if len(remaining) > 0:
            moves.append('Add')
        if len(current) > MIN_LENGTH_SUBSET:
            moves.append('Remove')
        if len(moves) == 0:
            return None
        if self.rng.choice(moves) == 'Add':
            return np.append(current, self.rng.choice(remaining))
        return np.delete(current, self.rng.integers(0, len(current)))

    def run(self):
        resumed = self.reload_weka_jvm
        if resumed == False:
            start_epoch = 0
            initial = self.rng.standard_normal(self.nFeatures) > 0
            if SET_MIN_LENGTH == True:
                initial = self.debug(initial)
            currentSubset = np.where(initial)[0]
            currentFitValue = self.evaluate(currentSubset, self.train_data)
            bestSubset = currentSubset.copy()
            bestFitValue = currentFitValue
            self.temperature = self.initTemp
        else:
            start_epoch = self.current_generation
            currentSubset = np.array(self.currentSubset)
            currentFitValue = self.currentFitValue
            bestSubset = np.array(self.bestSubset)
            bestFitValue = self.bestFitValue
        self.reload_weka_jvm = False
        for epoch in range(start_epoch, self.epochs):
            if epoch % MAX_ITERATIONS_JVM_WEKA == 0 and epoch > 1 and resumed == False:
                self.current_generation = epoch
                self.currentSubset = currentSubset
                self.currentFitValue = currentFitValue
                self.bestSubset = bestSubset
                self.bestFitValue = bestFitValue
                self.reload_weka_jvm = True
                self.printF('Ask for reload weka jvm')
                break
            resumed = False
            if self.temperature < self.min_temp:
                self.printF(f'Temperature {self.temperature:.2e} below {self.min_temp}. Termination condition met')
                break
            candidate = self.perturb(currentSubset)
            if candidate is None:
                self.printF('No feasible perturbation left. Termination condition met')
                break
            newFitValue = self.evaluate(candidate, self.train_data)
            if newFitValue > currentFitValue or self.rng.random() < self.acceptance_probability(currentFitValue, newFitValue):
                currentSubset = candidate
                currentFitValue = newFitValue
                if currentFitValue > bestFitValue:
                    self.printF('Global improvement in metric from {:8.4f} to {:8.4f} '
                        .format(bestFitValue, currentFitValue) + ' - Best subset updated')
                    bestFitValue = currentFitValue
                    bestSubset = currentSubset.copy()
            self.temperature = self.alpha * self.temperature
            parsedSubset = [self.original_indexes[i] for i in bestSubset]
            writterHistoric(f'{self.path_historic}/{self.name}_{self.type_of}.csv', {'epoch': epoch, 'gen': self.epochs, 'subset': str(parsedSubset), 'merit': bestFitValue})
            self.printF(f'Epoch: {epoch}/{self.epochs} - T: {self.temperature:.2e} - Best Score: {bestFitValue} - Length: {len(currentSubset)} - Evals: {self.n_evaluations}')
        if self.reload_weka_jvm:
            return
        self.bestScoreValue = bestFitValue
        self.bestSubset = list(bestSubset)
        self.bestFitValue = bestFitValue
        return self.bestSubset
