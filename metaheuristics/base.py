"""
Base Metaheuristic Module
=========================

This module provides the foundational classes and utilities for all metaheuristic
algorithms in the Metaheuristics-Mammaprint project.

Core Components:
---------------
1. **Metaheuristic Base Class**: Abstract base for all optimization algorithms
2. **Configuration Constants**: Global settings for algorithm behavior
3. **Utility Functions**: Helper functions for logging and data management

Global Constants:
----------------
MAX_ITERATIONS_JVM_WEKA : int
    Maximum iterations before reloading Weka JVM (default: 50)
    Prevents memory issues with long-running Java processes

SET_MIN_LENGTH : bool
    Enforce minimum feature subset length (default: True)
    Ensures biological/statistical validity of feature subsets

MIN_LENGTH_SUBSET : int
    Minimum number of features in valid subsets (default: 10)
    Prevents overfitting and ensures robust model performance

Author: ASP-ML Team
License: MIT
Version: 1.0.0
"""

import os

# ============================================================================
# GLOBAL CONFIGURATION CONSTANTS
# ============================================================================

MAX_ITERATIONS_JVM_WEKA = 50
"""Maximum iterations before reloading Weka JVM to prevent memory issues."""

SET_MIN_LENGTH = True
"""Enable enforcement of minimum feature subset length constraint."""

MIN_LENGTH_SUBSET = 10
"""Minimum required number of features in any valid subset."""

def writterHistoric(path: str, data: dict):
    """
    Write algorithm execution history to CSV file.
    
    Logs the progress of metaheuristic algorithms including generation number,
    epoch, selected feature subset, and objective function merit value.
    
    Parameters:
    ----------
    path : str
        Path to the CSV history file. Creates file if it doesn't exist.
    data : dict
        Dictionary containing execution data with keys:
        - 'gen': Total number of generations
        - 'epoch': Current epoch/iteration number
        - 'subset': List or string representation of selected features
        - 'merit': Objective function value/fitness score
    
    File Format:
    -----------
    CSV with columns: Gen, Epoch, Subset, Merit
    - Gen: Total algorithm generations
    - Epoch: Current iteration
    - Subset: Selected feature indices
    - Merit: Objective function score
    
    Example:
    -------
    >>> data = {
    ...     'gen': 100,
    ...     'epoch': 45,
    ...     'subset': '[0, 5, 10, 15, 20]',
    ...     'merit': 0.8523
    ... }
    >>> writterHistoric('results/GA_MI.csv', data)
    """
    if os.path.exists(path) == False:
        with open(path, 'w') as file:
            file.write('Gen,Epoch,Subset,Merit\n')
    else:
        with open(path, 'a') as file:
            file.write(f'{data["gen"]},{data["epoch"]},{data["subset"]},{data["merit"]}\n')

class Metaheuristic:
    """
    Abstract base class for all metaheuristic optimization algorithms.
    
    This class provides the common interface and basic functionality that all
    metaheuristic algorithms (GA, DE, PSO, ACO, SA, GWO) must implement.
    
    Attributes:
    ----------
    historic : list
        History of objective function values across iterations.
        Used for convergence analysis and visualization.
    
    bestScoreValue : float
        Best objective function value found during optimization.
        Represents the quality of the optimal feature subset.
    
    bestSubset : list
        Indices of features in the best subset found.
        This is the primary output of the optimization process.
    
    Methods:
    -------
    getHistoric()
        Returns the complete optimization history.
    
    getBestSubset()
        Returns the optimal feature subset indices.
    
    getTotalFeatures()
        Returns the number of features in the best subset.
    
    Notes:
    -----
    - All subclasses must implement: initialize(), run(), setDataSet()
    - Common pattern: Population-based iterative improvement
    - Tracks convergence through historic attribute
    - Handles dataset preprocessing and feature indexing
    
    Example:
    -------
    >>> class MyAlgorithm(Metaheuristic):
    ...     def __init__(self, data, labels, **params):
    ...         Metaheuristic.__init__(self)
    ...         self.data = data
    ...         self.labels = labels
    ...     
    ...     def run(self):
    ...         # Algorithm implementation
    ...         pass
    """
    def __init__(self):
        self.historic = []
        self.bestScoreValue = 0
        self.bestSubset = []
    
    def getHistoric(self):
        """
        Get the complete optimization history.
        
        Returns:
        -------
        list
            List of objective function values for each iteration.
            Useful for convergence plots and performance analysis.
        """
        return self.historic
    
    def getBestSubset(self):
        """
        Get the optimal feature subset found by the algorithm.
        
        Returns:
        -------
        list
            Indices of features in the best subset.
            These indices correspond to columns in the original dataset.
        """
        return self.bestSubset
    
    def getTotalFeatures(self):
        """
        Get the number of features in the best subset.
        
        Returns:
        -------
        int
            Total count of selected features.
            Useful for analyzing feature reduction effectiveness.
        """
        return len(list(self.bestSubset))
