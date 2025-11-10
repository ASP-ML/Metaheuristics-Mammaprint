"""
Gray Wolf Optimizer (GWO) for Feature Selection
===============================================

This module implements the Gray Wolf Optimizer algorithm adapted for feature
selection in the Mammaprint cancer classification project.

GWO is inspired by the social hierarchy and hunting behavior of gray wolves,
using alpha, beta, and delta wolves to guide the search process.

Algorithm Overview:
------------------
1. Initialize random wolf pack positions (feature subsets)
2. Identify three best wolves (alpha, beta, delta)
3. For each iteration:
   a. Update positions of all wolves based on leaders
   b. Adjust exploration/exploitation via coefficient a
   c. Evaluate fitness and update alpha, beta, delta
4. Return alpha wolf's position as best feature subset

Key Features:
------------
- Hierarchical leadership structure (alpha, beta, delta, omega)
- Adaptive exploration/exploitation balance
- Continuous to binary conversion via sigmoid/threshold
- Boundary handling for invalid positions
- Linear decay of coefficient a from 2 to 0

Mathematical Foundation:
-----------------------
Position update equation:
X(t+1) = X_p(t) - A · D
where D = |C · X_p(t) - X(t)|
A = 2a · r1 - a
C = 2 · r2
a: linearly decreases from 2 to 0

References:
----------
Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
Advances in engineering software, 69, 46-61.

Author: ASP-ML Team
License: MIT
Version: 1.0.0
"""

import numpy as np
import random
import time
from .base import Metaheuristic


class GrayWolfAlgorithm(Metaheuristic): 
    """
    Gray Wolf Optimizer for binary feature selection.
    
    Implements GWO with hierarchical wolf pack structure adapted for
    discrete feature selection problems.
    
    Attributes:
    ----------
    search_agents : int
        Number of wolves in the pack (population size)
    dim : int
        Dimensionality (number of features)
    lb : float or list
        Lower bound(s) for feature values (0 for binary)
    ub : float or list
        Upper bound(s) for feature values (1 for binary)
    fitnessFunction : callable
        Objective function to evaluate feature subsets
    epochs : int
        Maximum number of iterations
    """
    def __init__(self, data, labels, search_agents, max_iter, fitnessFunction):
        """
        Initialize Gray Wolf Optimizer for feature selection.
        
        Parameters:
        ----------
        data : pd.DataFrame or np.ndarray
            Feature matrix without target labels
        labels : pd.Series or np.ndarray
            Target class labels
        search_agents : int
            Number of wolves in the pack (typically 5-30)
        max_iter : int
            Maximum number of iterations
        fitnessFunction : callable
            Function to evaluate feature subset quality
        
        Notes:
        -----
        - Larger pack size improves exploration but increases computation
        - Alpha wolf represents best solution found
        - Beta and delta wolves guide search towards promising regions
        - Coefficient 'a' linearly decreases to balance exploration/exploitation
        """
        Metaheuristic.__init__(self)

        self.lb = 0
        self.ub = data.shape[1]
        self.dim = data.shape[1]
        self.train_data = data
        self.labels = labels
        self.search_agents = search_agents
        self.epochs = max_iter
        self.fitnessFunction = fitnessFunction
        self.name = 'GWA'
    
    def initialize(self):
        self.dim = self.train_data.shape[1]
        self.ub = self.train_data.shape[1]
        self.historic = []
        
    def runProcess(self, q):
        bestSubset = self.run()
        q.put((bestSubset, self.historic, self.name, self.epochs, self.bestScoreValue))
    
    def run(self):
        Alpha_pos = np.zeros(self.dim)
        Alpha_score = float(0)
        
        Beta_pos = np.zeros(self.dim)
        Beta_score = float(0)
        
        Delta_pos = np.zeros(self.dim)
        Delta_score = float(0)

        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim
        
        # Initialize the positions of search agents
        Positions = np.zeros((self.search_agents, self.dim))
        for i in range(self.dim):
            Positions[:, i] = np.random.uniform(0, 1, self.search_agents) * (self.ub[i] - self.lb[i]) + self.lb[i]

        timerStart = time.time() 
        # Main loop
        for l in range(0, self.epochs):
            for i in range(0, self.search_agents):
                
                # Return back the search agents that go beyond the boundaries of the search space
                for j in range(self.dim):
                    Positions[i, j] = np.clip(Positions[i, j], self.lb[j], self.ub[j])

                # Calculate objective function for each search agent
                fitness = self.fitnessFunction(np.where(Positions[i, :] > 0.49999)[0], self.train_data, self.labels)
                
                # Update Alpha, Beta, and Delta
                if fitness > Alpha_score:
                    Alpha_score = fitness  # Update alpha
                    Alpha_pos = Positions[i, :].copy()
                
                
                if (fitness < Alpha_score and fitness > Beta_score):
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i, :].copy()
                
                
                if (fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score): 
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i, :].copy()

            a = 2 - l * ((2) / self.epochs)  # a decreases linearly from 2 to 0
            
            # Update the Position of search agents including omegas
            for i in range(0, self.search_agents):
                for j in range(0, self.dim):     
                            
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
                    
                    A1 = 2 * a * r1 - a  # Equation (3.3)
                    C1 = 2 * r2  # Equation (3.4)
                    
                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])  # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1 * D_alpha  # Equation (3.6)-part 1
                            
                    r1 = random.random()
                    r2 = random.random()
                    
                    A2 = 2 * a * r1 - a  # Equation (3.3)
                    C2 = 2 * r2  # Equation (3.4)
                    
                    D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])  # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2       
                    
                    r1 = random.random()
                    r2 = random.random() 
                    
                    A3 = 2 * a * r1 - a  # Equation (3.3)
                    C3 = 2 * r2  # Equation (3.4)
                    
                    D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])  # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3             
                    
                    Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
                    
            self.historic.append(Alpha_score)

            if l % 1 == 0:
                print(['At iteration ' + str(l) + ' the best fitness is ' + str(Alpha_score)])
                print('alpha:', np.where(Alpha_pos > 0.5)[0])
        
        timerEnd = time.time()
        print('Completed in', (timerEnd - timerStart))
        self.bestScoreValue = Alpha_score
        self.bestSubset = np.where(Alpha_pos > 0.49999)[0]
        return self.bestSubset

    def setDataSet(self, train_data, labels):
        self.train_data = train_data
        self.labels = labels
        print(f'Set train dataset: {self.name} - {self.train_data.shape}')
