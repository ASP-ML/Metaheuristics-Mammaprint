"""
Particle Swarm Optimization (PSO) for Feature Selection
=======================================================

This module implements Particle Swarm Optimization using the NiaPy library,
adapted for feature selection in the Mammaprint project.

PSO is inspired by the social behavior of bird flocking and fish schooling,
where particles explore the solution space by following their own best position
and the swarm's global best position.

Algorithm Overview:
------------------
1. Initialize swarm of particles with random positions and velocities
2. For each iteration:
   a. Evaluate fitness of all particles
   b. Update personal best (pbest) for each particle
   c. Update global best (gbest) of entire swarm
   d. Update velocities using inertia, cognitive, and social components
   e. Update particle positions
3. Return global best position as optimal feature subset

Key Features:
------------
- Velocity-based position updates
- Inertia weight for exploration/exploitation balance
- Cognitive component (personal best memory)
- Social component (swarm global best)
- NiaPy integration for robust implementation
- Binary conversion via threshold (x > 0.5)

Mathematical Foundation:
-----------------------
Velocity update:
v(t+1) = w·v(t) + c1·r1·(pbest - x(t)) + c2·r2·(gbest - x(t))

Position update:
x(t+1) = x(t) + v(t+1)

where:
- w: inertia weight
- c1, c2: cognitive and social coefficients
- r1, r2: random values in [0,1]

References:
----------
Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings
of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.

NiaPy documentation: https://niapy.org/

Author: ASP-ML Team
License: MIT
Version: 1.0.0
"""

import numpy as np
from niapy.task import OptimizationType, Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization


class PSOProblem(Problem):
    """
    Problem definition for PSO feature selection using NiaPy framework.
    
    Wraps the feature selection problem in NiaPy's Problem interface,
    handling fitness evaluation and binary feature conversion.
    
    Attributes:
    ----------
    X_train : np.ndarray
        Training feature matrix
    y_train : np.ndarray
        Training labels
    objectiveFunction : callable
        Fitness function for feature subset evaluation
    alpha : float
        Weight parameter for fitness calculation (default: 0.99)
    historic : list
        History of fitness values across iterations
    """
    def __init__(self, X_train, y_train, objectiveFunction, alpha=0.99):
        """
        Initialize PSO problem for feature selection.
        
        Parameters:
        ----------
        X_train : np.ndarray
            Training data matrix (n_samples × n_features)
        y_train : np.ndarray
            Training labels
        objectiveFunction : callable
            Function to evaluate feature subset quality
        alpha : float, optional
            Weight for balancing accuracy vs feature count (default: 0.99)
        
        Notes:
        -----
        - Dimension equals number of features
        - Bounds set to [-10, 10] for continuous optimization
        - Binary features extracted via thresholding at 0.5
        """
        super().__init__(dimension=X_train.shape[1], lower=-10, upper=10)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.objectiveFunction = objectiveFunction
        self.name = 'PSO'
        self.historic = []
    
    def initialize(self):
        self.historic = []
    
    def setDataSet(self, data, labels):
        self.X_train = data
        self.y_train = labels

    def _evaluate(self, x):
        features = x > 0.5
        num_selected = features.sum()
        if num_selected == 0:
            return 1.0
        features = [i for i in range(len(features)) if features[i] == True]
        mi = self.objectiveFunction(features, self.X_train, self.y_train)
        self.historic.append(mi)
        return mi


class PSO:
    """
    PSO wrapper class for feature selection optimization.
    
    Coordinates PSO algorithm execution using NiaPy's ParticleSwarmOptimization
    implementation with custom problem definition.
    
    Attributes:
    ----------
    task : niapy.task.Task
        NiaPy optimization task configuration
    pso : niapy.algorithms.basic.ParticleSwarmOptimization
        PSO algorithm instance from NiaPy
    problem : PSOProblem
        Feature selection problem definition
    epochs : int
        Maximum number of iterations
    bestScoreValue : float
        Best fitness value found
    bestSubset : list
        Indices of best feature subset
    
    Methods:
    -------
    run()
        Execute PSO optimization and return best feature subset
    setDataSet(data, labels)
        Update training data for new optimization run
    """
    def __init__(self, task, pso, problem, of):
        """
        Initialize PSO algorithm wrapper.
        
        Parameters:
        ----------
        task : niapy.task.Task
            NiaPy task defining optimization parameters
        pso : niapy.algorithms.basic.ParticleSwarmOptimization
            PSO algorithm instance
        problem : PSOProblem
            Feature selection problem definition
        of : callable
            Objective function for fitness evaluation
        
        Notes:
        -----
        - NiaPy handles swarm initialization and updates automatically
        - Task configuration controls termination criteria
        - Problem._evaluate() called for each particle evaluation
        """
        self.task = task
        self.pso = pso
        self.problem = problem
        self.epochs = task.max_iters
        self.name = 'PSO'
        self.bestScoreValue = 0
        self.of = of
        self.bestSubset = []
    
    def initialize(self):
        self.bestSubset = []
    
    def setDataSet(self, data, labels):
        self.problem = PSOProblem(data, labels, self.of)
        self.task = Task(self.problem, max_iters=self.epochs, optimization_type=OptimizationType.MAXIMIZATION)
        self.pso = ParticleSwarmOptimization(data.shape[1], 1234)
        
        print(f'Set train dataset: {self.name} - {data.shape}')
    
    def getBestSubset(self):
        return self.bestSubset
    
    def runProcess(self, q):
        bestSubset = self.run()
        q.put((bestSubset, self.getHistoric(), self.name, self.epochs, self.bestScoreValue))
    
    def run(self):
        print('Running PSO from wrapper object')
        best_subset, best_fitness_value = self.pso.run(self.task)
        self.bestScoreValue = best_fitness_value
        self.bestSubset = np.where(best_subset > 0.49999)[0]
        return self.bestSubset
    
    def getHistoric(self):
        return self.task.convergence_data()[1]
