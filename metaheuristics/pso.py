import numpy as np
from niapy.task import OptimizationType, Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization


class PSOProblem(Problem):
    def __init__(self, X_train, y_train, objectiveFunction, alpha=0.99):
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
    """PSO Algorithm"""
    def __init__(self, task, pso, problem, of):
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
