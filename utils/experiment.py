

import ast
from datetime import date, datetime
from multiprocessing import Queue
from multiprocessing.dummy import Process
import os
import pandas as pd
import numpy as np
import shared_imports as si

class ExecutorWrapper:
    def __init__(self, executor, future) -> None:
        self.executor = executor
        self.future = future
    def stopExecutor(self):
        self.executor.shutdown()
class MetaheuristicStateWrapper:
    def __init__(self, metaheuristic) -> None:
        self.metaheuristic = metaheuristic
        self.done = False
        self.process_assigned = False
    def setParams(self, params):
        self.params = params

    def getParams(self):
        return self.params
    
    def setProcessAssigned(self, value):
        self.process_assigned = value

    def processAssigned(self):
        return self.process_assigned
    
    def isDone(self):
        return self.done
    
    def setDone(self, state):
        print('>> Done with ', self.metaheuristic.name, '_', self.metaheuristic.type_of, 'State:', state)
        self.done = state

class Experiment:

    def __init__(self, n_steps: int, number_metas: int) -> None:
        self.data_experiment = [[[] for _ in range(number_metas)] for _ in range(n_steps)]
        print('DATA EXPERIMENT: ', self.data_experiment)
        print('LEN ONE STEP: ', len(self.data_experiment[0]))
        self.header_columns = ['Model', 'Settings', 'Name', 'Generation', 'Subset', 'Length', 'Merit']

    def addDataByMeta(self, params: dict, step_index: int):
        experiment_index = params['experiment_index']
        self.data_experiment[step_index][experiment_index].append([params['name'], params['settings'], 
                                                             params['of_name'], params['iterations'], 
                                                             str(params['best_subset']),
                                                             str(len(params['best_subset'])),
                                                              params['best_score_value']])
    def getBestByObjectiveFunction(self, index_step, get_df: bool = False, add_key_column: bool = True):
        merge = np.concatenate(self.data_experiment[index_step])
        df = pd.DataFrame(merge, columns= self.header_columns)
        df['Merit'] = df['Merit'].astype('float64')
        df = df.loc[df.groupby(['Model', 'Name']).Merit.idxmax(),['Model','Generation', 'Subset', 'Length', 'Name', 'Settings', 'Merit']]
        if add_key_column:
            df['key'] = df['Model'] + '_' + df['Name']

        if get_df:
            return df.to_numpy(), df 
        else:
            return df.to_numpy()
    def getHistoricByStep(self, index_step):
        merge = np.concatenate(self.data_experiment[index_step])
        df = pd.DataFrame(merge, columns= self.header_columns)
        return df
    def getLastDataEntry(self):
        return self.getBestByObjectiveFunction(len(self.data_experiment) - 1)
    
    def checkMetaheuristicStates(self, metaheuristic_dict):
        i = 0
        done = True
        for key in metaheuristic_dict:
            print('CHECK mh finish i: ', i)
            if metaheuristic_dict[key].isDone() == False:
                done = False
                break
            i += 1
        print('DONE: ', done)
        return done
    
def runExperimentInOrder(mh_array, data, labels, runPipelineClassifiers = True):
    subset_results = []
    processList = []
    date_today = date.today().strftime("%m_%d_%Y")
    time = str(datetime.now().strftime("%H_%M_%S"))
    path = 'results/' + date_today + '/' + time
    bestHistoric = []
    historicLen = []
    if os.path.exists(path) == False:
        os.makedirs(path)

    q = Queue()
    for i in mh_array:
        p = Process(target = i.runProcess, args = (q,))
        processList.append((p, i))
        p.start()

    for process, mhObject in processList:
        print('Waiting for process: ', process.name)
        process.join()
        bestSubset, historic, name, _, bestValue = q.get()
        subset_results.append((bestSubset, historic, name))
        file = si.createLogFile(name, path)
        if len(bestSubset) == 0:
            bestSubset = [1]
            print('>> Detectado caso 0')
        if runPipelineClassifiers == True:
            bestClassifier = si.classifiers.pipeline_classifiers(bestSubset, data, labels, 1, date_today, time, file, name)
            file.close()
            bestHistoric.append(np.append(bestClassifier, [name]))
            historicLen.append(np.append(bestClassifier, [name, len(bestSubset)]))
        else:
            bestHistoric.append(np.append(bestValue, [name]))
            historicLen.append(np.append(bestValue, [name, len(bestSubset)]))
    
    return bestHistoric, subset_results, historicLen 
        
        
def runExperiment(mh_array, data, labels, runPipelineClassifiers = True):
    print('>> Ejecutando algoritmos metaheurísticos seleccionados')
    subset_results = []
    epochs = []
    times = 10
    processList = []
    q = Queue()
    for i in mh_array:
        p = Process(target= i.runProcess, args=(q,))
        processList.append((p, i))
        p.start()
    for process, mhObject in processList:
        print('Waiting for process: ', process.name)
        process.join()
        subset_results.append(q.get())
        epochs.append(mhObject.epochs)

    print('Subset: ', subset_results)    
    date_today = date.today().strftime("%m_%d_%Y")
    time = str(datetime.now().strftime("%H_%M_%S"))
    path = 'results/' + date_today + '/' + time
    
    if os.path.exists(path) == False:
        os.makedirs(path)
    bestHistoric = []
    historicToGetMax = []
    if runPipelineClassifiers == True:
        for i in subset_results:
            if i[2] + '_log.csv' not in os.listdir(path):
                file = open(path + '/' + i[2] +  '_log.csv', 'a')
                file.write('k-fold,ACC,PRE,MSE,REC,AUC,F1,MCC\n')
            else:
                file = open(path + '/' + i[2] +  '_log.csv', 'a') 
            bestClassifier = si.classifiers.pipeline_classifiers(i[0], data, labels, times, date_today, time, file, i[2])
            bestHistoric.append(np.append(bestClassifier, [i[2]]))
            historicToGetMax.append( np.append(bestClassifier, [i[2], len(i[0])]))
            file.close()
    else:
        for i in subset_results:
            historicToGetMax.append(np.append(i[4], [i[2], len(i[0])]))
    subsets_report = [[i[1], np.array(i[0]), str(np.array(i[0]).shape)] if type(i[0]) == list else [i[1], i[0], str(i[0].shape)] for i in subset_results]
    np.savetxt(path + '/subsets_' + time +'.csv', subsets_report, delimiter=',', fmt='%s')
    #Saving the experiment    
    nameExperiment = ''
    if runPipelineClassifiers == True:
        nameExperiment = 'exp_' + date_today + '_' + time + '.txt'
        expDf = pd.DataFrame(columns=['Classifier', 'ACC MEAN', 'ACC DESV', 'PRE MEAN', 'PRE DESV', 'MSE MEAN', 'MSE DESV', 'REC MEAN', 'REC DESV', 'AUC MEAN', 'AUC DESV', 'F1 MEAN', 'F1 DESV', 'MCC MEAN','MCC DESV', 'Metaheuristic'], data= bestHistoric)
        expDf.to_csv(path + '/' + nameExperiment, index=False)
    #Starting to plot
    max_epochs = max(epochs)
    x = np.arange(max_epochs)
    for i in subset_results:
        historic = np.array(i[1])
        if historic.shape[0] != x.shape[0]:
            historic = np.pad(historic, (0, abs(x.shape[0] - historic.shape[0])))
        si.plt.plot(x, historic, label = i[2] + ' Epochs: ' + str(i[3]))
    si.plt.xlabel('Epochs')
    si.plt.ylabel('Objective function value')
    si.plt.legend()
    si.plt.savefig(path + '/fig_'+time)
    return bestHistoric, path + '/' + nameExperiment, subset_results, np.array(historicToGetMax)

def getMetasByObjectiveFunction(data, labels, mh_array, of_array, generations, results_generated = np.array([]), path_report = None):
    local_data = data
    dic_metas = {}
    keys = np.array([])
    indexes = [i for i in range(data.shape[1])]
    if len(results_generated) > 0:
        keys = results_generated[:, -1]
    experiment_index = 0
    for i in mh_array:
        population = int(round(data.shape[1] / 2, 0))
        for j in of_array:
            key = i + '_' + j
            if np.isin(key, keys): 
                index = np.where(key == keys)[0][0]
                subset = ast.literal_eval(results_generated[index][2])
                print(f'>> {key}: SUBSET LOADED - {subset}')
                local_data = si.classifiers.generateSubBM(subset, data)
                indexes = subset
            if local_data.shape[1] > 10:
                if i == 'SA':
                    dic_metas[key] = MetaheuristicStateWrapper(si.SimulatedAnnealing(experiment_index, local_data, labels, 40, generations, local_data.shape[1], 0, 0.8, si.ofunctions.allObjectiveFunctionWeka, j, indexes, path_report)) 
                    
                if i == 'GA':
                    dic_metas[key] = MetaheuristicStateWrapper(si.GeneticAlgorithm(experiment_index, local_data,labels,generations, population, local_data.shape[1], 0.20, 20, si.ofunctions.allObjectiveFunctionWeka, j, indexes, path_report)) 

                if i == 'DE':
                    print('Loading new data for de: ', local_data, local_data.shape)
                    dic_metas[key] = MetaheuristicStateWrapper(si.DifferentialEvolution(experiment_index, local_data,labels, population, local_data.shape[1], generations, 0.5, 0.45, si.ofunctions.allObjectiveFunctionWeka, j, indexes, path_report)) 
            else:
                print('>> The subset length is less than the minimum length required - ', key)
        experiment_index = experiment_index + 1
    return dic_metas
