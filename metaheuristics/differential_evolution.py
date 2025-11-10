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
Version: 1.0.0
"""

import numpy as np
import logging
import weka.core.jvm as jvm
from .base import Metaheuristic, writterHistoric, MAX_ITERATIONS_JVM_WEKA, SET_MIN_LENGTH, MIN_LENGTH_SUBSET


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
    def __init__(self, experiment_index, data, labels, pop_size, n_features, epochs, mutationFactor, crossOverFactor, objectiveFunction, type_of, indexes, path_historic):
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
        
        Notes:
        -----
        - Population initialized with uniform random values in [0, 1]
        - Binary feature selection via thresholding continuous values
        - Automatically enforces minimum subset length if SET_MIN_LENGTH=True
        """
        Metaheuristic.__init__(self)
        self.pop_size = pop_size
        self.pop = np.random.rand(pop_size, n_features)
        self.nFeatures = n_features
        self.objectiveFunction = objectiveFunction
        self.epochs = epochs
        self.mutationFactor = mutationFactor
        self.crossOverFactor = crossOverFactor
        self.name = 'DE'
        self.type_of = type_of
        self.experiment_index = experiment_index
        self.reload_weka_jvm = False
        self.current_best_score = 0
        self.current_generation = 0
        self.current_best_subset = []
        self.prevObjValue = 0
        self.path_historic = path_historic
        self.obj_all = None
        self.setDataSet(data, labels, indexes)
        if SET_MIN_LENGTH == True:
            self.pop = self.debug(self.pop)
            print('POP: ', self.pop)
    
    def getParams(self):
        info_dict = {
            'name': self.name,
            'of_name': self.type_of,
            'iterations': self.epochs,
            'settings': 'Pop: ' + str(self.pop_size) + ' N features: ' + str(self.nFeatures) + ' MF: ' + str(self.mutationFactor) + ' COF: ' + str(self.crossOverFactor),
            'best_subset': self.bestSubset,
            'best_score_value': self.bestScoreValue,
            'experiment_index': self.experiment_index,
            'current_best_subset': self.current_best_subset,
            'current_generation': self.current_generation,
            'current_best_score': self.current_best_score,
            'reload_weka_jvm': self.reload_weka_jvm,
            'population': self.pop,
            'prevObjValue': self.prevObjValue,
            'obj_all': self.obj_all
        }
        return info_dict
    
    def setParams(self, params):
        self.pop = params['population']
        self.reload_weka_jvm = params['reload_weka_jvm']
        self.current_best_score = params['current_best_score']
        self.current_generation = params['current_generation']
        self.current_best_subset = params['current_best_subset']
        self.prevObjValue = params['prevObjValue']
        self.obj_all = params['obj_all']
    
    def initialize(self):
        self.nFeatures = self.train_data.shape[1]
        self.pop = np.random.rand(self.pop_size, self.nFeatures)
        if SET_MIN_LENGTH == True:
            self.pop = self.debug(self.pop)
        if self.nFeatures == 1:
            self.pop[0][0] = 0.75
        self.historic = []
    
    def set_new_iterations(self, iterations):
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
        self.printF('Executing debug')
        arrayFixed = []
        for i in population:
            check = np.where(i > 0.499999)[0].shape[0]
            if check < (MIN_LENGTH_SUBSET):
                while np.where(i > 0.499999)[0].shape[0] < (MIN_LENGTH_SUBSET):
                    random_int = np.random.randint(0, self.nFeatures)
                    i[random_int] = np.random.uniform(0.5, 1)
            arrayFixed.append(i)
        return np.array(arrayFixed)
    
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
        p = np.random.rand(self.nFeatures)
        rnbr = np.random.randint(self.nFeatures, size=1)[0]
        trial = [target[i] if (p[i] > self.crossOverFactor and i != rnbr) else mutated[i] for i in range(self.nFeatures)]
        return trial
    
    def runProcess(self):
        jvm.start(packages=True, logging_level=logging.INFO)
        _ = self.run()
        jvm.stop()
        return self.getParams()
    
    def printF(self, msg):
        print(f'>> {self.name} - {self.type_of}: ', msg)
    
    def run(self):
        # First the algorithm computes the objectiveFunction value for the initial population
        reload_weka = False
        bestObjValue = 0
        if self.reload_weka_jvm == False:
            epoch = 0
            obj_all = [self.objectiveFunction(np.where(i > 0.499999)[0], self.train_data, self.labels, self.type_of) for i in self.pop]
            bestChromosome = self.pop[np.argmax(obj_all)]
            tempo_chromosome = self.pop[np.argmax(obj_all)]
            lenTempo = np.where(tempo_chromosome > 0.499999)[0].shape[0]
            if lenTempo >= 9:
                bestChromosome = tempo_chromosome
                prevObjValue = bestObjValue
                nFeaturesSelected = np.where(bestChromosome > 0.499999)[0].shape[0]
        else:
            epoch = self.current_generation
            prevObjValue = self.prevObjValue
            obj_all = self.obj_all
            bestChromosome = self.current_best_subset
            reload_weka = True
        # print('POPULATION LOADED: ', self.pop)
        nFeaturesSelected = np.where(bestChromosome > 0.499999)[0].shape[0]
        bestObjValue = max(obj_all)
        prevObjValue = bestObjValue
        self.reload_weka_jvm = False
        # Starting with the epochs
        for epoch in range(epoch, self.epochs):
            if epoch % MAX_ITERATIONS_JVM_WEKA == 0 and epoch != 1 and epoch != 0 and reload_weka == False:
                self.prevObjValue = prevObjValue
                self.reload_weka_jvm = True
                self.current_generation = epoch
                self.obj_all = obj_all
                self.current_best_subset = bestChromosome
                self.printF('Ask for reload weka jvm')
                break
            for chromosome in range(self.pop_size):
                # Choose three candidates a, b and c that are not the current one
                candidates = [candidate for candidate in range(self.pop_size) if candidate != chromosome]
                a, b, c = self.pop[np.random.choice(candidates, 3, replace=False)]
                # Perform a mutation
                mutated = self.mutation([a, b, c], self.mutationFactor)
                mutated = np.clip(mutated, 0, 1)
                # Perform a crossover using the constant of crossOver and the totalOfFeatures as dimensions
                trial = self.crossOver(mutated=mutated, target=self.pop[chromosome])
                trial = np.around(trial, decimals=8)
                # Compute the objective value for the trial vector
                # objTarget = self.objectiveFunction(self.pop[chromosome], self.train_data, self.labels, entropyDataSet)
                objTarget = self.objectiveFunction(np.where(self.pop[chromosome] > 0.4999)[0], self.train_data, self.labels, self.type_of)
                objTrial = self.objectiveFunction(np.where(trial > 0.49999)[0], self.train_data, self.labels, self.type_of)
                # Perform a selection
                if objTrial > objTarget:
                    # Replace the targetVector with the trialVector
                    if SET_MIN_LENGTH == True and (np.where(trial > 0.49999)[0].shape[0] >= (MIN_LENGTH_SUBSET)):
                        self.pop[chromosome] = trial
                        obj_all[chromosome] = objTrial
            # Find the best after each iteration
            bestObjValue = max(obj_all)
            if bestObjValue > prevObjValue:
                tempo_chromosome = self.pop[np.argmax(obj_all)]
                lenTempo = np.where(tempo_chromosome > 0.499999)[0].shape[0]
                if lenTempo >= 9:
                    bestChromosome = tempo_chromosome
                    prevObjValue = bestObjValue
                    nFeaturesSelected = np.where(bestChromosome > 0.499999)[0].shape[0]
                else:
                    print('Length dont allowed')
            parsedSubset = [self.original_indexes[i] for i in list(np.where(np.array(bestChromosome) > 0.499999)[0])]
            writterHistoric(f'{self.path_historic}/{self.name}_{self.type_of}.csv', {'epoch': epoch, 'gen': self.epochs, 'subset': str(parsedSubset), 'merit': prevObjValue})
            self.printF(f'Epoch: {epoch}/{self.epochs} - Best Score: {prevObjValue} - Length: {nFeaturesSelected}')
            reload_weka = False
        self.bestScoreValue = prevObjValue
        print('Best chromsome: ', bestChromosome)
        print('WHERE: ', np.where(np.array(bestChromosome) > 0.499999))
        self.bestSubset = [self.original_indexes[i] for i in list(np.where(np.array(bestChromosome) > 0.499999)[0])]
        print('Self best subset: ', self.bestSubset, len(self.bestSubset))
        return self.bestSubset
