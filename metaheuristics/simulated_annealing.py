import numpy as np
import logging
import weka.core.jvm as jvm
from .base import Metaheuristic, writterHistoric, MAX_ITERATIONS_JVM_WEKA, SET_MIN_LENGTH, MIN_LENGTH_SUBSET


class SimulatedAnnealing(Metaheuristic):
    """Simulated Annealing Algorithm"""
    def __init__(self, experiment_index, data, labels, initTemp, iterations, nFeatures, beta, alpha, objectiveFunction, type_of, indexes, path_historic):
        """:param data: dataset
           :param labels: dataset labels
           :param initTemp: init temp of metal bar
           :param iterations: number of generations to repeat the algorithm
           :param nFeatures: the number of features to consider in the algorithm
           :param beta: is used for acceptance probability
           :param alpha: is used for the reduction of temperature"""
        Metaheuristic.__init__(self)
        self.initTemp = initTemp
        self.epochs = iterations
        self.objectiveFunction = objectiveFunction
        self.nFeatures = nFeatures
        self.alpha = alpha
        self.type_of = type_of
        self.name = 'SA'
        self.experiment_index = experiment_index
        self.reload_weka_jvm = False
        self.current_generation = 0
        self.previousFitValue = 0
        self.bestSubset = None
        self.bestFitValue = 0
        self.path_historic = path_historic
        self.setDataSet(data, labels, indexes)
    
    def set_new_iterations(self, iterations):
        self.initialize()
        self.epochs = iterations
        self.bestSubset = []
        self.bestScoreValue = 0
        self.reload_weka_jvm = False
    
    def setParams(self, params):
        self.reload_weka_jvm = params['reload_weka_jvm']
        self.previousFitValue = params['previousFitValue']
        self.bestSubset = params['bestSubset']
        self.bestFitValue = params['bestFitValue']
        self.initTemp = params['initTemp']
        self.current_generation = params['current_generation']

    def setExperimentIndex(self, experiment_index):
        self.experiment_index = experiment_index

    def getParams(self):
        info_dict = {
            'name': self.name,
            'of_name': self.type_of,
            'iterations': self.epochs,
            'settings': 'T0: ' + str(self.initTemp) + ' N features: ' + str(self.nFeatures) + ' Alpha: ' + str(self.alpha),
            'best_subset': [self.original_indexes[i] for i in list(self.bestSubset)],
            'best_score_value': self.bestScoreValue,
            'experiment_index': self.experiment_index,
            'current_generation': self.current_generation,
            'reload_weka_jvm': self.reload_weka_jvm,
            'previousFitValue': self.previousFitValue,
            'bestSubset': self.bestSubset,
            'bestFitValue': self.bestFitValue,
            'initTemp': self.initTemp
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
        self.previousFitValue = 0
        self.bestSubset = None
        self.bestFitValue = 0
    
    def runProcess(self):
        jvm.start(packages=True, logging_level=logging.INFO)
        _ = self.run()
        jvm.stop()
        return self.getParams()
    
    def printF(self, msg):
        print(f'>> {self.name} - {self.type_of}: ', msg)
    
    def debug(self, subset):
        self.printF('Executing debug')
        subset_copy = subset.copy()
        check = np.where(subset_copy > 0.499999)[0].shape[0]
        if check < (MIN_LENGTH_SUBSET):
            while np.where(subset_copy > 0.499999)[0].shape[0] < (MIN_LENGTH_SUBSET):
                random_int = np.random.randint(0, self.nFeatures)
                subset_copy[random_int] = np.random.uniform(0.5, 1)
        return np.array(subset_copy)
    
    def run(self):
        reload_weka = False
        if self.reload_weka_jvm == False:
            epoch = 0
            currentSubSet = np.random.rand(self.nFeatures)
            if SET_MIN_LENGTH == True:
                currentSubSet = self.debug(currentSubSet)
            currentSubSet = np.where(currentSubSet > 0.499999)[0]
            bestSubSet = currentSubSet
        else:
            epoch = self.current_generation
            currentSubSet = self.bestSubset
            bestSubSet = self.bestSubset
            reload_weka = True

        previousFiValue = self.objectiveFunction(currentSubSet, self.train_data, self.labels, self.type_of)
        bestFitValue = previousFiValue
        subSetGenerated = set()
        self.reload_weka_jvm = False
        for epoch in range(epoch, self.epochs):
            self.printF(str(currentSubSet))
            if epoch % MAX_ITERATIONS_JVM_WEKA == 0 and epoch != 1 and epoch != 0 and reload_weka == False:
                self.bestSubset = bestSubSet
                self.previousFitValue = previousFiValue
                self.bestFitValue = bestFitValue
                self.current_generation = epoch
                self.reload_weka_jvm = True
                self.printF('Ask for reload weka jvm')
                break
            reload_weka = False
            if self.initTemp < 0.001:
                self.printF(f'Temperature {self.initTemp} below threshold. Termination condition met')
                break
            remaningFlag = False
            while True:
                # Decide what type of pertubation choose
                # if len(currentSubSet) >= MIN_LENGTH_SUBSET and self.nFeatures >= 10:
                #     move = 'Remove'
                # elif len(currentSubSet) <= 10: # Not to go below 10 features
                #     move = np.random.choice(['Add', 'Replace'])
                # else:
                if len(currentSubSet) > MIN_LENGTH_SUBSET:
                    move = np.random.choice(['Add', 'Replace', 'Remove'])
                else:
                    move = np.random.choice(['Add', 'Replace'])
                remaningFeatures = [index for index in range(self.nFeatures) if index not in currentSubSet]
                perturbedSubSet = currentSubSet.copy()
                if len(remaningFeatures) == 0:
                    remaningFlag = True
                    break
                # Generation the perturbation
                if move == 'Add':
                    perturbedSubSet = np.append(perturbedSubSet, np.random.choice(remaningFeatures))
                elif move == 'Replace':
                    perturbedSubSet = np.delete(perturbedSubSet, np.random.randint(0, len(perturbedSubSet)))
                    perturbedSubSet = np.append(perturbedSubSet, np.random.choice(remaningFeatures))
                elif len(currentSubSet) > MIN_LENGTH_SUBSET:
                    perturbedSubSet = np.delete(perturbedSubSet, np.random.randint(0, len(perturbedSubSet)))
                    self.printF('REMOVED')

                if set(perturbedSubSet) in subSetGenerated or len(perturbedSubSet) < 10:
                    self.printF('Subset already visited!')
                    self.printF('PERTURBED: ' + str(perturbedSubSet))
                else:
                    subSetGenerated.add(frozenset(perturbedSubSet))
                    break
            if remaningFlag == True:
                self.printF('No more features to make a perturbation')
                break
            if (len(perturbedSubSet) < 10):
                self.printF('Perturbedset dont have the requerided length')
                break
            newFitValue = self.objectiveFunction(perturbedSubSet, self.train_data, self.labels, self.type_of)
            if len(currentSubSet) >= 10:
                if newFitValue > previousFiValue:
                    currentSubSet = perturbedSubSet.copy()
                    if newFitValue > bestFitValue:
                        self.printF('Global improvement in metric from {:8.4f} to {:8.4f} '
                            .format(bestFitValue, newFitValue) + ' - Best subset updated')
                        bestFitValue = newFitValue
                        
                        bestSubSet = list(perturbedSubSet.copy())
                else:
                    randomValue = np.random.uniform()
                    difference = previousFiValue - newFitValue
                    accept_prob = np.exp(-(difference) / self.initTemp)

                    if randomValue < accept_prob:
                        previousFiValue = newFitValue
                        currentSubSet = list(perturbedSubSet.copy())
            else:
                print('Subset length less that min: ', len(currentSubSet))
                break
            self.initTemp = self.alpha * self.initTemp
            writterHistoric(f'{self.path_historic}/{self.name}_{self.type_of}.csv', {'epoch': epoch, 'gen': self.epochs, 'subset': str(bestSubSet), 'merit': bestFitValue})
            self.printF(f'Epoch: {epoch}/{self.epochs} - Best Score: {bestFitValue} - Length: {len(currentSubSet)}')
            
        self.bestScoreValue = bestFitValue
        self.bestSubset = bestSubSet
        return bestSubSet
