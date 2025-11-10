import numpy as np
import random
import logging
import multiprocessing
import concurrent.futures
import weka.core.jvm as jvm
from .base import Metaheuristic, writterHistoric, MAX_ITERATIONS_JVM_WEKA, SET_MIN_LENGTH, MIN_LENGTH_SUBSET


class GeneticAlgorithm(Metaheuristic):
    """Genetic Algorithm, https://www.kaggle.com/code/tanmayunhale/genetic-algorithm-for-feature-selection/notebook
    """
    def __init__(self, experiment_index, data, labels, generations, population_size, length_chromosome, mutationRange, n_parents, objectiveFunction, type_of, indexes, path_historic):
        """
        :param data: data without labels
        :param labels: dataset labels
        :param generations: number of generation to repeat the algorithm
        :param population_size: size of the population size
        :param length_chromosome: length of chromosome or number of features
        :param mutationRange: mutation factor
        :param n_parents: number of parents for the next population
        :param objectiveFunction: the objective function to optimize"""

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
        self.initialize()

    def setExperimentIndex(self, experiment_index):
        self.experiment_index = experiment_index

    def set_new_iterations(self, iterations):
        self.initialize()
        self.epochs = iterations
        self.bestSubset = []
        self.bestScoreValue = 0
        self.current_best_score = 0
        self.current_generation = 0
        self.current_best_subset = []
        self.reload_weka_jvm = False
    
    def setParams(self, params):
        self.population = params['population']
        self.reload_weka_jvm = params['reload_weka_jvm']
        self.current_best_score = params['current_best_score']
        self.current_generation = params['current_generation']
        self.current_best_subset = params['current_best_subset']
    
    def getParams(self):
        info_dict = {
            'name': self.name,
            'of_name': self.type_of,
            'iterations': self.epochs,
            'settings': 'Pop: ' + str(self.population_size) + ' LC: ' + str(self.length_chromosome) + ' MR: ' + str(self.mutationRange) + ' NP: ' + str(self.n_parents),
            'best_subset': self.bestSubset,
            'best_score_value': self.bestScoreValue,
            'experiment_index': self.experiment_index,
            'current_best_subset': self.current_best_subset,
            'current_generation': self.current_generation,
            'current_best_score': self.current_best_score,
            'reload_weka_jvm': self.reload_weka_jvm,
            'population': self.population
        }
        return info_dict
    
    def initialize(self):
        self.population = []
        self.name = 'GA'
        self.length_chromosome = self.data.shape[1]
        self.population = []
        for _ in range(self.population_size):
            chromosome = np.random.rand(self.data.shape[1]) > 0.49999999
            self.population.append(chromosome)  
        self.population = np.array(self.population)
        if SET_MIN_LENGTH == True:
           self.population = self.debug(self.population)
        self.historic = []
        self.score_list = None
        self.score_list_sorted = None
        self.current_best_score = 0
        self.current_generation = 0
        self.current_best_subset = []
        self.reload_weka_jvm = False
        print(f'{self.name}-{self.type_of}: Population: ', len(self.population), len(self.population[0]))

    def setDataSet(self, data, labels, original_indexes):
        self.data = data
        self.labels = labels
        self.original_indexes = original_indexes

    def selectionFA(self):
        indexes_et = self.score_list_sorted[0:self.elite_children,]
        self.score_list = np.delete(self.score_list, indexes_et, 0)
        self.score_list_sorted = np.argsort(self.score_list)[::-1]
        elite_children = self.population[indexes_et]
        self.population = np.delete(self.population, indexes_et, 0)
        return elite_children

    def selection(self, population, n_parents, score_indexes, score_list):
        populationNextGen = []
        for _ in range(n_parents):
            winner = self.tournament(score_indexes, score_list)
            populationNextGen.append(population[winner])
        return populationNextGen
    
    def tournament(self, score_sort_indexes, score_list, size=2):
        random_indexes = np.random.randint(0, score_sort_indexes.shape[0], size)
        parents_indexes = score_sort_indexes[random_indexes]
        obj_function_values = [score_list[i] for i in parents_indexes]
        sorted_indexes = np.argsort(obj_function_values)
        return parents_indexes[sorted_indexes][1]
    
    def crossOverFA(self):
        array = []
        for _ in range(0, self.pop_cross_over):
            winner_1_index = self.tournament()
            winner_2_index = self.tournament()
            winner_1 = self.population[winner_1_index]
            winner_2 = self.population[winner_2_index]
            array.append(np.array(np.logical_xor(winner_1, winner_2), dtype=np.int))
        return np.array(array)
    
    def crossOver(self, popSel):
        popNextGen = popSel
        for i in range(0, len(popSel), 2):
            child1, child2 = popNextGen[i], popNextGen[i + 1]
            popNextGen.append(np.concatenate((child1[:(len(child1))//2], child2[(len(child1)//2):])))
        return np.array(popNextGen)
    
    def debug(self, population):
        arrayFixed = []
        for i in population:
            check = np.sum(i, where=[True])
            if check < MIN_LENGTH_SUBSET:
                while np.sum(i, where=[True]) < MIN_LENGTH_SUBSET:
                    random_int = np.random.randint(0, self.length_chromosome)
                    i[random_int] = True
            arrayFixed.append(i)
        return arrayFixed
        
    def mutation(self, pop_after_cross):   
        mutation_range = int(self.mutationRange * self.data.shape[1])
        pop_next_gen = []
        for n in range(0, len(pop_after_cross)):
            chromo = pop_after_cross[n]
            rand_posi = [] 
            for i in range(0, mutation_range):
                pos = np.random.randint(0, self.data.shape[1] - 1)
                rand_posi.append(pos)
            for j in rand_posi:
                chromo[j] = not chromo[j]  
            pop_next_gen.append(chromo)
        return pop_next_gen
    
    def runProcess(self):
        if self.reload_weka_jvm == False:
            self.initialize()
        jvm.start(packages=True, logging_level=logging.INFO)
        _ = self.run()
        jvm.stop()
        return self.getParams()
    
    def printF(self, msg):
        print(f'>> {self.name} - {self.type_of}: ', msg)
    
    def wrapperObjective(self, features):
        n_procs = 1
        ctx = multiprocessing.get_context('spawn')
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_procs, mp_context=ctx)
        futures = []
        for _ in range(n_procs):
            futures.append(executor.submit(self.objectiveFunction, features, self.data, self.labels, self.type_of))
        result = None
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
        executor.shutdown()
        return result
    
    def run(self):
        tempo = 0
        bestSubset = []
        bestScore = 0
        reload_weka = False
        if self.reload_weka_jvm == True:
            bestSubset = self.current_best_subset
            bestScore = self.current_best_score
            tempo = self.current_generation
            reload_weka = True
        self.reload_weka_jvm = False
        while tempo < self.epochs:
            self.current_generation = tempo
            if tempo % MAX_ITERATIONS_JVM_WEKA == 0 and tempo != 1 and tempo != 0 and reload_weka == False:
                self.current_best_subset = bestSubset
                self.current_best_score = bestScore
                self.reload_weka_jvm = True
                self.printF('Ask for reload weka jvm')
                break
            score_list = []
            for chromosome in self.population:
                features = [i for i in range(len(chromosome)) if chromosome[i] == True]
                score = self.objectiveFunction(features, self.data, self.labels, self.type_of)
                score_list.append(score)
            score_list_indexes = np.argsort(np.array(score_list))[::-1]
            # if bestScore < score_list[score_list_indexes[0]]:
            bestSubset = self.population[score_list_indexes[0]]
            bestScore = score_list[score_list_indexes[0]]
            popAfterSel = self.selection(self.population, self.n_parents, score_list_indexes, score_list)
            if SET_MIN_LENGTH == True:
                popAfterSel = self.debug(popAfterSel)
            popAfterCross = self.crossOver(popAfterSel)
            if SET_MIN_LENGTH == True:
                popAfterCross = self.debug(popAfterCross)
            self.population = self.mutation(popAfterCross)
            if SET_MIN_LENGTH == True:
                self.population = self.debug(self.population)
            tempo += 1
            reload_weka = False
            parsedSubset = [self.original_indexes[i] for i in range(len(bestSubset)) if bestSubset[i] == True]
            self.printF(f'Epoch: {tempo}/{self.epochs} - Best Score: {bestScore} - Length: {len(parsedSubset)}')
            writterHistoric(f'{self.path_historic}/{self.name}_{self.type_of}.csv', {'epoch': tempo, 'gen': self.epochs, 'subset': str(parsedSubset), 'merit': bestScore})
        if self.reload_weka_jvm:
            return
        self.bestScoreValue = bestScore
        self.bestSubset = [self.original_indexes[i] for i in range(len(bestSubset)) if bestSubset[i] == True]
        return self.bestSubset
