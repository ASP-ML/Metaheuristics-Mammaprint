import numpy as np
import random
import time
from .base import Metaheuristic


class GrayWolfAlgorithm(Metaheuristic): 
    """Gray Wolf Algorithm"""
    def __init__(self, data, labels, search_agents, max_iter, fitnessFunction):
        Metaheuristic.__init__(self)
        """ :param data: dataset without labels
            :param labels: dataset labels
            :param search_agents: the number of wolfs for search the best objetive value
            :param max_iter: max iterations
            :param fitnessFunction: the objective function to optimize"""
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
