"""
Genetic Algorithm (GA) Metaheuristic for Feature Selection
==========================================================

This module implements a Genetic Algorithm specifically designed for feature
selection tasks in the Mammaprint cancer classification project.

Genetic Algorithms are population-based evolutionary algorithms that mimic
natural selection through selection, crossover, and mutation operators.

Algorithm Overview:
------------------
1. Initialize random population of binary chromosomes (feature subsets) from a
   Gaussian distribution (a gene is selected when its draw is > 0)
2. For each generation:
   a. Evaluate fitness of all individuals
   b. Select parents using binary tournament selection
   c. Create offspring through one-point crossover (random cut point)
   d. Apply bit-flip mutation (each gene flips with probability mutationRange)
   e. The mutated offspring form the new population (constant size)
3. Return best feature subset found across all generations

Key Features:
------------
- Binary chromosome encoding (1=selected, 0=not selected)
- Tournament selection for parent selection
- One-point crossover with a random cut point, two children per pair
- Per-gene bit-flip mutation
- The best chromosome found is saved; optional elitism through N_ELITE
- JVM reload handling for Weka objective functions

References:
----------
Holland, J. H. (1992). Genetic algorithms. Scientific american, 267(1), 66-73.
Adapted from: https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection

Author: ASP-ML Team
License: MIT
Version: 1.1.0
"""

import numpy as np
import logging
import weka.core.jvm as jvm
from .base import (Metaheuristic, writterHistoric, stream_key,
                   MAX_ITERATIONS_JVM_WEKA, SET_MIN_LENGTH, MIN_LENGTH_SUBSET)

N_ELITE = 0
"""Number of best chromosomes copied unchanged into the next generation (0 = paper: P(t+1) is the mutated offspring)."""


class GeneticAlgorithm(Metaheuristic):
    """
    Genetic Algorithm for binary feature selection optimization.

    Implements classical GA with tournament selection, crossover, and mutation
    operators adapted for feature selection with minimum subset constraints.

    Attributes:
    ----------
    generations : int
        Number of evolutionary generations (epochs)
    population_size : int
        Number of individuals in population (constant across generations)
    length_chromosome : int
        Length of binary chromosome (equals number of features)
    mutationRange : float
        Mutation probability per bit
    n_parents : int
        Number of parents selected for reproduction each generation
    population : np.ndarray
        Current population of binary chromosomes (population_size × length_chromosome)
    """
    def __init__(self, experiment_index, data, labels, generations, population_size, length_chromosome, mutationRange, n_parents, objectiveFunction, type_of, indexes, path_historic, seed=None):
        """
        Initialize Genetic Algorithm for feature selection.

        Parameters:
        ----------
        experiment_index : int
            Unique identifier for experimental run tracking
        data : pd.DataFrame or np.ndarray
            Feature matrix without target variable
        labels : pd.Series or np.ndarray
            Target class labels
        generations : int
            Number of evolutionary generations to run
        population_size : int
            Size of population (typically 50-200 individuals)
        length_chromosome : int
            Length of binary chromosome (number of features)
        mutationRange : float
            Probability of bit-flip mutation per gene (commonly 1 / length_chromosome)
        n_parents : int
            Number of parents selected each generation (mating pool size, >= 2)
        objectiveFunction : callable
            Fitness function evaluating feature subset quality
        type_of : str
            Identifier for objective function type
        indexes : array-like
            Feature indices mapping to original dataset
        path_historic : str
            CSV file path for logging optimization history
        seed : int or None
            Base seed of the private random generator (None = not reproducible)

        Notes:
        -----
        - Higher mutation rates increase exploration but may disrupt convergence
        - Tournament size affects selection pressure (larger = more elitist)
        - Population size should scale with problem complexity
        """

        Metaheuristic.__init__(self)
        self.name = 'GA'
        self.setDataSet(data, labels, indexes)
        self.mutationRange = mutationRange
        self.n_parents = n_parents
        self.epochs = generations
        self.objectiveFunction = objectiveFunction
        self.population = []
        self.population_size = population_size
        self.length_chromosome = length_chromosome
        self.type_of = type_of
        self.experiment_index = experiment_index
        self.reload_weka_jvm = False
        self.path_historic = path_historic
        self.set_seed(seed, stream_key(f'{self.name}_{self.type_of}'), length_chromosome)
        self.initialize()

    def setExperimentIndex(self, experiment_index):
        self.experiment_index = experiment_index

    def set_new_iterations(self, iterations):
        # Each restart budget gets its own reproducible random stream
        self.reseed(iterations)
        self.initialize()
        self.epochs = iterations
        self.bestSubset = []
        self.bestScoreValue = 0

    def setParams(self, params):
        self.population = params['population']
        self.reload_weka_jvm = params['reload_weka_jvm']
        self.current_best_score = params['current_best_score']
        self.current_generation = params['current_generation']
        self.current_best_subset = params['current_best_subset']
        self.n_evaluations = params['n_evaluations']
        self.rng.bit_generator.state = params['rng_state']

    def getParams(self):
        info_dict = {
            'name': self.name,
            'of_name': self.type_of,
            'iterations': self.epochs,
            'settings': 'Pop: ' + str(self.population_size) + ' LC: ' + str(self.length_chromosome) + ' MR: ' + str(self.mutationRange) + ' NP: ' + str(self.n_parents) + ' Evals: ' + str(self.n_evaluations),
            'best_subset': self.bestSubset,
            'best_score_value': self.bestScoreValue,
            'experiment_index': self.experiment_index,
            'current_best_subset': self.current_best_subset,
            'current_generation': self.current_generation,
            'current_best_score': self.current_best_score,
            'reload_weka_jvm': self.reload_weka_jvm,
            'population': self.population,
            'n_evaluations': self.n_evaluations,
            'rng_state': self.rng.bit_generator.state
        }
        return info_dict

    def initialize(self):
        self.name = 'GA'
        self.length_chromosome = self.data.shape[1]
        population = self.rng.standard_normal((self.population_size, self.length_chromosome)) > 0
        if SET_MIN_LENGTH == True:
           population = self.debug(population)
        self.population = population
        self.historic = []
        self.score_cache = {}
        self.current_best_score = 0
        self.current_generation = 0
        self.current_best_subset = None
        self.reload_weka_jvm = False
        self.n_evaluations = 0
        print(f'{self.name}-{self.type_of}: Population: ', len(self.population), len(self.population[0]))

    def setDataSet(self, data, labels, original_indexes):
        self.data = data
        self.labels = labels
        self.original_indexes = original_indexes

    def fitness(self, chromosome):
        """Objective value of a chromosome; duplicates are looked up instead of re-evaluated."""
        key = chromosome.tobytes()
        if key not in self.score_cache:
            self.score_cache[key] = self.evaluate(np.where(chromosome)[0], self.data)
        return self.score_cache[key]

    def selection(self, population, n_parents, score_list):
        """Fill the mating pool with binary tournament winners."""
        winners = [self.tournament(score_list) for _ in range(n_parents)]
        return np.array(population)[winners].copy()

    def tournament(self, score_list, size=2):
        """Index of the fittest of `size` individuals drawn at random (maximization)."""
        contenders = self.rng.integers(0, len(score_list), size)
        return contenders[int(np.argmax(np.asarray(score_list)[contenders]))]

    def crossOver(self, parents, n_offspring):
        """One-point crossover with a random cut point; each pair of parents yields two children."""
        offspring = []
        while len(offspring) < n_offspring:
            i, j = self.rng.choice(len(parents), 2, replace=False)
            cut = self.rng.integers(1, self.length_chromosome)
            offspring.append(np.concatenate((parents[i][:cut], parents[j][cut:])))
            offspring.append(np.concatenate((parents[j][:cut], parents[i][cut:])))
        return np.array(offspring[:n_offspring])

    def debug(self, population):
        """Repair chromosomes selecting fewer than MIN_LENGTH_SUBSET features."""
        population = np.array(population, dtype=bool)
        min_length = min(MIN_LENGTH_SUBSET, self.length_chromosome)
        for chromosome in population:
            while chromosome.sum() < min_length:
                chromosome[self.rng.integers(0, self.length_chromosome)] = True
        return population

    def mutation(self, offspring):
        """Bit-flip mutation: every gene flips independently with probability mutationRange."""
        flips = self.rng.random(offspring.shape) < self.mutationRange
        return np.logical_xor(offspring, flips)

    def runProcess(self):
        if self.reload_weka_jvm == False:
            self.initialize()
        jvm.start(packages=True, logging_level=logging.INFO)
        _ = self.run()
        jvm.stop()
        return self.getParams()

    def printF(self, msg):
        print(f'>> {self.name} - {self.type_of}: ', msg)

    def run(self):
        resumed = self.reload_weka_jvm
        if resumed == True:
            bestSubset = self.current_best_subset
            bestScore = self.current_best_score
            tempo = self.current_generation
            # The cache does not survive the process restart
            self.score_cache = {}
        else:
            bestSubset = None
            bestScore = -np.inf
            tempo = 0
        self.reload_weka_jvm = False
        while tempo < self.epochs:
            self.current_generation = tempo
            if tempo % MAX_ITERATIONS_JVM_WEKA == 0 and tempo > 1 and resumed == False:
                self.current_best_subset = bestSubset
                self.current_best_score = bestScore
                self.reload_weka_jvm = True
                self.printF('Ask for reload weka jvm')
                break
            resumed = False
            score_list = [self.fitness(chromosome) for chromosome in self.population]
            generation_best = int(np.argmax(score_list))
            # Keep the best chromosome found across all generations
            if bestSubset is None or score_list[generation_best] > bestScore:
                bestSubset = self.population[generation_best].copy()
                bestScore = score_list[generation_best]
            popAfterSel = self.selection(self.population, self.n_parents, score_list)
            popAfterCross = self.crossOver(popAfterSel, self.population_size - N_ELITE)
            offspring = self.mutation(popAfterCross)
            if SET_MIN_LENGTH == True:
                offspring = self.debug(offspring)
            # With N_ELITE > 0 the best chromosome so far survives unchanged (elitism)
            self.population = np.vstack([np.tile(bestSubset, (N_ELITE, 1)), offspring])
            tempo += 1
            parsedSubset = [self.original_indexes[i] for i in np.where(bestSubset)[0]]
            self.printF(f'Epoch: {tempo}/{self.epochs} - Best Score: {bestScore} - Length: {len(parsedSubset)} - Evals: {self.n_evaluations}')
            writterHistoric(f'{self.path_historic}/{self.name}_{self.type_of}.csv', {'epoch': tempo, 'gen': self.epochs, 'subset': str(parsedSubset), 'merit': bestScore})
        if self.reload_weka_jvm:
            return
        self.bestScoreValue = bestScore
        self.bestSubset = [] if bestSubset is None else [self.original_indexes[i] for i in np.where(bestSubset)[0]]
        return self.bestSubset
