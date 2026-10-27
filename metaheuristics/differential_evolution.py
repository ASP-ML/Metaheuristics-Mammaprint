"""
Differential Evolution (DE) Metaheuristic for Feature Selection
==============================================================

This module implements the Differential Evolution algorithm adapted for feature
selection tasks in the Mammaprint dataset analysis.

Differential Evolution is a population-based stochastic optimization algorithm
that uses mutation, crossover, and selection operators to evolve candidate solutions.

Algorithm Overview:
------------------
1. Initialize random population of feature subsets
2. For each generation:
   a. Create mutant vectors through differential mutation
   b. Apply crossover to create trial vectors
   c. Evaluate fitness of trial vectors
   d. Select better solutions for next generation
3. Return best feature subset found

Key Features:
------------
- Binary representation for feature selection
- Adaptive mutation and crossover strategies
- JVM reload handling for long-running Weka evaluations
- Minimum subset length enforcement
- Convergence tracking and history logging

References:
----------
Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient
heuristic for global optimization over continuous spaces. Journal of global
optimization, 11(4), 341-359.

Author: ASP-ML Team
License: MIT
Version: 1.1.0
"""

import numpy as np
import logging
import weka.core.jvm as jvm
from .base import (Metaheuristic, writterHistoric, selected_features, stream_key,
                   MAX_ITERATIONS_JVM_WEKA, SET_MIN_LENGTH, MIN_LENGTH_SUBSET)


class DifferentialEvolution(Metaheuristic):
    """
    Differential Evolution algorithm for feature selection optimization.

    This class implements DE/rand/1/bin variant adapted for binary feature
    selection problems with support for Weka-based objective functions.

    Attributes:
    ----------
    pop_size : int
        Population size (number of candidate solutions)
    pop : np.ndarray
        Current population matrix (pop_size × n_features)
    nFeatures : int
        Total number of features in dataset
    epochs : int
        Maximum number of generations
    mutationFactor : float
        Differential weight F ∈ [0, 2], typically 0.5-0.8
    crossOverFactor : float
        Crossover probability CR ∈ [0, 1], typically 0.1-0.9
    objectiveFunction : callable
        Fitness evaluation function for feature subsets
    """
    def __init__(self, experiment_index, data, labels, pop_size, n_features, epochs, mutationFactor, crossOverFactor, objectiveFunction, type_of, indexes, path_historic, seed=None):
        """
        Initialize Differential Evolution for feature selection.

        Parameters:
        ----------
        experiment_index : int
            Unique identifier for this experimental run
        data : pd.DataFrame or np.ndarray
            Feature matrix without target labels
        labels : pd.Series or np.ndarray
            Target class labels for supervised learning
        pop_size : int
            Number of individuals in population (typically 5-10 × n_features)
        n_features : int
            Total number of features available for selection
        epochs : int
            Maximum number of generations to evolve
        mutationFactor : float
            Differential weight F, controls mutation step size (0.4-1.0)
        crossOverFactor : float
            Crossover probability CR, controls recombination (0.1-0.9)
        objectiveFunction : callable
            Function to evaluate feature subset quality
        type_of : str
            Name/type of objective function being used
        indexes : array-like
            Feature indices mapping to original dataset columns
        path_historic : str
            File path to save optimization history CSV
        seed : int or None
            Base seed of the private random generator (None = not reproducible)

        Notes:
        -----
        - Population initialized with uniform random values in [0, 1]
        - Binary feature selection via thresholding continuous values at 0.5
        - Automatically enforces minimum subset length if SET_MIN_LENGTH=True
        """
        Metaheuristic.__init__(self)
        self.pop_size = pop_size
        self.nFeatures = n_features
        self.objectiveFunction = objectiveFunction
        self.epochs = epochs
        self.mutationFactor = mutationFactor
        self.crossOverFactor = crossOverFactor
        self.name = 'DE'
        self.type_of = type_of
        self.experiment_index = experiment_index
        self.reload_weka_jvm = False
        self.current_generation = 0
        self.path_historic = path_historic
        self.obj_all = None
        self.setDataSet(data, labels, indexes)
        self.set_seed(seed, stream_key(f'{self.name}_{self.type_of}'), n_features)
        self.pop = self.rng.random((pop_size, n_features))
        if SET_MIN_LENGTH == True:
            self.pop = self.debug(self.pop)

    def getParams(self):
        info_dict = {
            'name': self.name,
            'of_name': self.type_of,
            'iterations': self.epochs,
            'settings': 'Pop: ' + str(self.pop_size) + ' N features: ' + str(self.nFeatures) + ' MF: ' + str(self.mutationFactor) + ' COF: ' + str(self.crossOverFactor) + ' Evals: ' + str(self.n_evaluations),
            'best_subset': self.bestSubset,
            'best_score_value': self.bestScoreValue,
            'experiment_index': self.experiment_index,
            'current_generation': self.current_generation,
            'reload_weka_jvm': self.reload_weka_jvm,
            'population': self.pop,
            'obj_all': self.obj_all,
            'n_evaluations': self.n_evaluations,
            'rng_state': self.rng.bit_generator.state
        }
        return info_dict

    def setParams(self, params):
        self.pop = params['population']
        self.reload_weka_jvm = params['reload_weka_jvm']
        self.current_generation = params['current_generation']
        self.obj_all = params['obj_all']
        self.n_evaluations = params['n_evaluations']
        self.rng.bit_generator.state = params['rng_state']

    def initialize(self):
        self.nFeatures = self.train_data.shape[1]
        self.pop = self.rng.random((self.pop_size, self.nFeatures))
        if SET_MIN_LENGTH == True:
            self.pop = self.debug(self.pop)
        if self.nFeatures == 1:
            self.pop[0][0] = 0.75
        self.historic = []
        self.obj_all = None
        self.current_generation = 0
        self.reload_weka_jvm = False
        self.n_evaluations = 0

    def set_new_iterations(self, iterations):
        # Each restart budget gets its own reproducible random stream
        self.reseed(iterations)
        self.initialize()
        self.epochs = iterations
        self.bestSubset = []
        self.bestScoreValue = 0

    def setDataSet(self, data, labels, indexes):
        self.train_data = data
        self.labels = labels
        self.original_indexes = indexes
        print(f'Set train dataset: {self.name} - {self.train_data.shape}')

    def setExperimentIndex(self, experiment_index):
        self.experiment_index = experiment_index

    def debug(self, population):
        """Repair individuals selecting fewer than MIN_LENGTH_SUBSET features."""
        self.printF('Executing debug')
        population = np.array(population, dtype=float)
        min_length = min(MIN_LENGTH_SUBSET, self.nFeatures)
        for individual in population:
            while selected_features(individual).shape[0] < min_length:
                individual[self.rng.integers(0, self.nFeatures)] = self.rng.uniform(0.5, 1)
        return population

    def mutation(self, x, F):
        """Function to make the muttation of trial chromosome
            :param x: Trial vector
            :param F: mututation factor
        """
        return x[0] + F * (x[1] - x[2])

    def crossOver(self, mutated, target):
        """According to the crossover factor, the trial is composed using the mutated chromosome and the target chromosome
            :param mutated: mutated chromosome
            :param target: target chromosome"""
        p = self.rng.random(self.nFeatures)
        rnbr = self.rng.integers(self.nFeatures)
        take_mutated = p <= self.crossOverFactor
        take_mutated[rnbr] = True
        return np.where(take_mutated, mutated, target)

    def runProcess(self):
        jvm.start(packages=True, logging_level=logging.INFO)
        _ = self.run()
        jvm.stop()
        return self.getParams()

    def printF(self, msg):
        print(f'>> {self.name} - {self.type_of}: ', msg)

    def run(self):
        resumed = self.reload_weka_jvm
        if resumed == False:
            start_epoch = 0
            obj_all = [self.evaluate(selected_features(individual), self.train_data) for individual in self.pop]
        else:
            start_epoch = self.current_generation
            obj_all = list(self.obj_all)
        self.reload_weka_jvm = False
        # Greedy one-to-one selection never loses the population best, so the best
        # individual is always argmax(obj_all)
        best_index = int(np.argmax(obj_all))
        bestChromosome = self.pop[best_index].copy()
        bestObjValue = obj_all[best_index]
        for epoch in range(start_epoch, self.epochs):
            if epoch % MAX_ITERATIONS_JVM_WEKA == 0 and epoch > 1 and resumed == False:
                self.current_generation = epoch
                self.obj_all = obj_all
                self.reload_weka_jvm = True
                self.printF('Ask for reload weka jvm')
                break
            resumed = False
            # Trials are built from generation G; the survivors form generation G + 1
            next_pop = self.pop.copy()
            for chromosome in range(self.pop_size):
                # Choose three candidates a, b and c that are not the current one
                candidates = [candidate for candidate in range(self.pop_size) if candidate != chromosome]
                a, b, c = self.pop[self.rng.choice(candidates, 3, replace=False)]
                mutated = np.clip(self.mutation([a, b, c], self.mutationFactor), 0, 1)
                trial = np.around(self.crossOver(mutated=mutated, target=self.pop[chromosome]), decimals=8)
                trial_features = selected_features(trial)
                # Infeasible trials are rejected without spending an evaluation
                if SET_MIN_LENGTH == True and trial_features.shape[0] < MIN_LENGTH_SUBSET:
                    continue
                # The target fitness is already stored in obj_all (the objective is deterministic)
                objTrial = self.evaluate(trial_features, self.train_data)
                if objTrial > obj_all[chromosome]:
                    next_pop[chromosome] = trial
                    obj_all[chromosome] = objTrial
            self.pop = next_pop
            best_index = int(np.argmax(obj_all))
            if obj_all[best_index] > bestObjValue:
                bestObjValue = obj_all[best_index]
                bestChromosome = self.pop[best_index].copy()
            parsedSubset = [self.original_indexes[i] for i in selected_features(bestChromosome)]
            writterHistoric(f'{self.path_historic}/{self.name}_{self.type_of}.csv', {'epoch': epoch, 'gen': self.epochs, 'subset': str(parsedSubset), 'merit': bestObjValue})
            self.printF(f'Epoch: {epoch}/{self.epochs} - Best Score: {bestObjValue} - Length: {len(parsedSubset)} - Evals: {self.n_evaluations}')
        self.bestScoreValue = bestObjValue
        self.bestSubset = [self.original_indexes[i] for i in selected_features(bestChromosome)]
        print('Self best subset: ', self.bestSubset, len(self.bestSubset))
        return self.bestSubset
