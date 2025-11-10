"""
Shared imports module for Metaheuristics-Mammaprint project.

This module consolidates all commonly used imports following Python best practices.
Import categories are organized as: Standard Library → Third-party → Local packages.

Usage:
    from shared_imports import *
    # or
    from shared_imports import pd, np, plt, GeneticAlgorithm
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import ast
import gc
import logging
import multiprocessing
import os
import random
import time
from datetime import datetime, date
from multiprocessing import Process, Queue
import concurrent.futures
import ofunctions
import classifiers

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

# Data Science & Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metaheuristic Libraries
from niapy.task import OptimizationType, Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization

# Weka Integration
import weka.core.jvm as jvm

# =============================================================================
# LOCAL PACKAGE IMPORTS
# =============================================================================

# Metaheuristic Algorithms
from metaheuristics import (
    Metaheuristic,
    DifferentialEvolution,
    AntColony,
    Ant,
    interconnection,
    GeneticAlgorithm,
    GrayWolfAlgorithm,
    PSO,
    PSOProblem,
    SimulatedAnnealing,
    writterHistoric,
    MAX_ITERATIONS_JVM_WEKA,
    SET_MIN_LENGTH,
    MIN_LENGTH_SUBSET
)

# Utility Functions
from utils import *

# =============================================================================
# COMMONLY USED CONSTANTS AND CONFIGURATIONS
# =============================================================================

# Default configurations
DEFAULT_NUMBER_EXECUTORS = 2
DEFAULT_COLUMNS_LEGEND = ['Metric', 'Metaheuristic', 'Machine learning classifier']
DEFAULT_COLUMNS_LEGEND_2 = ['Metric', 'Metaheuristic']

# Default legend data for plots
DEFAULT_LEGEND_DATA = [
    ['MI - Mutual Information', 'GA - Genetic Algorithm', 'SVM - Support vector machine'],
    ['DMI - Diferential Mutual Information', 'PSO - Particle swarm optimization', 'KNN - k-Nearest neighbors'],
    ['PMI - Proportion Mutual Information', 'ABACO - Advanced binary ant colony', 'MLP - Multilayer perceptron'],
    ['', 'GWO - Gray wolf optimization', 'CART - Cart tree'],
    ['', 'SA - Simulated annealing', 'QDA - Quadratic discriminant analysis'],
    ['', 'DE - Differential Evolution', 'LDA - Linear discriminant analysis'],
    ['', '', 'NB - Naive bayes']
]

# =============================================================================
# UTILITY FUNCTIONS FOR COMMON OPERATIONS
# =============================================================================

def setup_logging(level=logging.INFO, log_file=None):
    """
    Setup logging configuration for the project.

    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *(logging.FileHandler(log_file) for _ in [log_file] if log_file)
        ]
    )

def initialize_jvm():
    """Initialize Weka JVM if not already initialized."""
    try:
        jvm.start(packages=True, logging_level=logging.INFO)
    except Exception as e:
        logging.warning(f"Failed to initialize JVM: {e}")

def cleanup_jvm():
    """Stop Weka JVM."""
    try:
        jvm.stop()
    except Exception as e:
        logging.warning(f"Failed to stop JVM: {e}")

# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "ASP-ML Team"
__description__ = "Shared imports module for Metaheuristics-Mammaprint project"

# Define what gets imported with "from shared_imports import *"
__all__ = [
    # Standard library
    'ast', 'gc', 'logging', 'multiprocessing', 'os', 'random', 'time',
    'datetime', 'date', 'Process', 'Queue', 'concurrent',

    # Third-party
    'pd', 'np', 'plt', 'OptimizationType', 'Task', 'Problem', 'ParticleSwarmOptimization', 'jvm',

    # Metaheuristics
    'Metaheuristic', 'DifferentialEvolution', 'AntColony', 'Ant', 'interconnection',
    'GeneticAlgorithm', 'GrayWolfAlgorithm', 'PSO', 'PSOProblem', 'SimulatedAnnealing',
    'writterHistoric', 'MAX_ITERATIONS_JVM_WEKA', 'SET_MIN_LENGTH', 'MIN_LENGTH_SUBSET',

    # Utilities
    'printAlgorithm', 'processDataSet', 'processDataUsingPath', 'Experiment',
    'ExecutorWrapper', 'MetaheuristicStateWrapper',

    # Project modules (may be None if not available)
    'ofunctions', 'metaheuristic', 'classifiers',

    # Constants
    'DEFAULT_NUMBER_EXECUTORS', 'DEFAULT_COLUMNS_LEGEND', 'DEFAULT_COLUMNS_LEGEND_2', 'DEFAULT_LEGEND_DATA',

    # Utility functions
    'setup_logging', 'initialize_jvm', 'cleanup_jvm'
]