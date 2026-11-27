import ast
import multiprocessing
import os
import time
import ofunctions
import pandas as pd
import numpy as np
import metaheuristic
import classifiers
from datetime import datetime
from datetime import date
import matplotlib.pyplot as plt
from niapy.task import OptimizationType
from multiprocessing import Process, Queue
import concurrent.futures
import gc
NUMBER_EXECUTORS = 2
columns_legend = ['Metric', 'Metaheuristic', 'Machine learning classifier']
columns_legend_2 = ['Metric', 'Metaheuristic']
legend_data = [['MI - Mutual Information', 'GA - Genetic Algorithm', 'SVM - Support vector machine'], 
               ['DMI - Diferential Mutual Information', 'PSO - Particle swarm optimization', 'KNN - k-Nearest neighbors'], 
               ['PMI - Proportion Mutual Information', 'ABACO - Advanced binary ant colony', 'MLP - Multilayer perceptron'],
               ['', 'GWO - Gray wolf optimization', 'CART - Cart tree'],
               ['', 'SA - Simulated annealing', 'QDA - Quadratic discriminant analysis'],
               ['', 'DE - Differential Evolution', 'LDA - Linear discriminant analysis'],
               ['', '', 'NB - Naive bayes'],]


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
        print('PARAMS: ', experiment_index)
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
        #print('DATA EXPERIMENT: ', np.array(self.data_experiment[index_step]).shape)
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
def printAlgorithm(name):
    print('Ha seleccionado: ', name)

def processDataSet(data_name):
    data = pd.read_csv('data/' + data_name)
    labels = data['label'].to_numpy(dtype=np.intp)
    data = data.drop(['label'], axis= 1).to_numpy()
    return data, labels
def processDataUsingPath(path):
    data = pd.read_csv(path)
    labels = data['label'].to_numpy()
    data = data.drop(['label'], axis= 1).to_numpy()
    return data, labels

def setTimesSCV():
    print('Ingrese el número de veces que desea ejecutar el Stratified K-Fold Cross Validation')
    while True: 
        try:
            k_times = int(input('>> Número de veces: '))
        except ValueError:
            print('El valor ingresado no es un número')
        if k_times > 0:
            print(f'>> Ha ingresado el siguiente valor: {k_times}')
            return k_times
        else:
            print('El número de generaciones debe ser mayor a cero')            

def setParameters(algorithm):
    print('Establezca los siguientes parámetros del algoritmo: ', algorithm)
    while True:
        try:
            gen = int(input('>> Generaciones: '))
        except ValueError:
            print('Debe ingresar un valor número mayor a cero')
            continue
        if gen > 0:
            print(f'>> Ha ingresado el siguiente valor: {gen}')
            return gen
        else:
            print('El número de generaciones debe ser mayor a cero')

def processSubset(features, data_name):
    data = pd.read_csv('data/' + data_name)
    labels = data['label'].to_numpy()
    data = data.drop(['label'], axis= 1).to_numpy()
    data = ofunctions.generateSubBM(features, data)
    return data, labels

def subsetByPath(features, path):
    data = pd.read_csv(path)
    labels = data['label'].to_numpy()
    data = data.drop(['label'], axis= 1).to_numpy()
    data = ofunctions.generateSubBM(features, data)
    return data, labels

def addRowsDataFrame(df, matrix):
    for i in matrix:
        df.loc[len(df.index)] = i

def selectObjectiveFunction():
    print('Funciones objetivo: \n1. Sumatoria de mutual information\n2. Diferencia de mutual information\n3. Proporción de mutual information\n4. SVM Mean AUC\n5. KNN Mean AUC')
    while True:
        try:
            option = int(input('>> Ingrese el número que identifica a la función objetivo: '))
        except ValueError:
            print('Debe ingresar un número')
            continue
        if option >= 1 and option <=5:
            print(f'>> Ha escogido la función {option}')
            return option
        else:
            print(f'Debe ingresar un valor entre 1 y 5')

def askAgain():
    print('**** Correr de nuevo el experimento tomando en cuenta el mejor subconjunto generado? ****')
    while True:
        try:
            option = int(input('1. Yes\n2. No\n0. Salir\n>>'))
        except ValueError:
            print('Debe ingresar un número')
            continue
        if option >= 0 and option <=2:
            print(f'>> Ha escogido la opción {option}')
            return option
        else:
            print(f'Debe ingresar un valor entre 0 y 2')
def runInOrder():
    print('**** Correr de nuevo el paso en orden ? ****')
    while True:
        try:
            option = int(input('1. Si\n2. No\n0. Salir\n>>'))
        except ValueError:
            print('Debe ingresar un número')
            continue
        if option >= 0 and option <=2:
            print(f'>> Ha escogido la opción {option}')
            return option
        else:
            print('Debe ingresar un valor entre 0 y 2')
def runNewStep():
    print('**** Correr nuevo paso tomando el cuenta:****')
    while True:
        try:
            option = int(input('1. AUC\n2. Manteniendo el orden\n0. Salir\n>>'))
        except ValueError:
            print('Debe ingresar un número')
            continue
        if option >= 0 and option <=2:
            print(f'>> Ha escogido la opción {option}')
            return option
        else:
            print(f'Debe ingresar un valor entre 0 y 2')
def generateResume(results):
    print('To generate the table')
    baseColumns = ['Method', 'MI']
    stepColumns = ['AUC', 'Length', 'MLC']
    dfList = []
    print('Results: ', results)
    expList = []
    for i in results:
        data = []
        print('i: ', i)
        nameExpDf = pd.DataFrame(columns = baseColumns)
        for j in i:
            data.append([j[9], j[16], j[0]])
            nameExpDf.loc[len(nameExpDf)] = [j[15], j[15]]
        expList.append(nameExpDf)
        dfList.append(pd.DataFrame(columns=stepColumns, data = data))

    
    experiments = list(range(len(results)))
    row_start = 2
    col_start = 3
    time = str(datetime.now().strftime("%H_%M_%S"))
    with pd.ExcelWriter(f'{time}.xlsx', engine='xlsxwriter') as writer:
        nameExpDf.to_excel(writer, sheet_name='Validation', startrow=2, startcol = 0)
        rowHeaderIndex = 0
        for rowHeader in expList:
            holdIndex = False
            if rowHeaderIndex == 0:
                holdIndex = True
            rowHeader.to_excel(writer, sheet_name='Validation', startrow=2, startcol = rowHeaderIndex, index = holdIndex)
            rowHeaderIndex += 6
        for i in dfList:
            print(i)
            i.to_excel(writer, sheet_name='Validation', startrow=row_start, startcol = col_start, index=False)
            col_start += 5
        # df_legend.to_excel(writer, sheet_name='Validation', startrow=nameExpDf.shape[0] + 5, index=False)
        workbook = writer.book
        merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})
        worksheet = writer.sheets['Validation']
        tempo = 3
        header_row = 2
        for i in experiments:
            worksheet.merge_range(header_row,tempo,header_row,tempo + 2,data= f'Step {i + 1}', cell_format = merge_format)
            tempo += 5
        print('Excel report generated: ', time)
def generateResumeWithoutPipeline(results):
    print('To generate the table')
    df_legend = pd.DataFrame(columns = columns_legend, data = legend_data)
    baseColumns = ['Method', 'MI']
    stepColumns = ['AUC', 'Length']
    dfList = []
    print('Results: ', results)
    expList = []
    for i in results:
        data = []
        print('i: ', i)
        nameExpDf = pd.DataFrame(columns = baseColumns)
        for j in i:
            data.append([j[0], j[2]])
            nameExpDf.loc[len(nameExpDf)] = [j[1], j[1]]
        expList.append(nameExpDf)
        dfList.append(pd.DataFrame(columns=stepColumns, data = data))

    
    experiments = list(range(len(results)))
    row_start = 2
    col_start = 3
    time = str(datetime.now().strftime("%H_%M_%S"))
    with pd.ExcelWriter(f'{time}.xlsx', engine='xlsxwriter') as writer:
        nameExpDf.to_excel(writer, sheet_name='Validation', startrow=2, startcol = 0)
        rowHeaderIndex = 0
        for rowHeader in expList:
            holdIndex = False
            if rowHeaderIndex == 0:
                holdIndex = True
            rowHeader.to_excel(writer, sheet_name='Validation', startrow=2, startcol = rowHeaderIndex, index = holdIndex)
            rowHeaderIndex += 6
        for i in dfList:
            print(i)
            i.to_excel(writer, sheet_name='Validation', startrow=row_start, startcol = col_start, index=False)
            col_start += 5
        # df_legend.to_excel(writer, sheet_name='Validation', startrow=nameExpDf.shape[0] + 5, index=False)
        workbook = writer.book
        merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})
        worksheet = writer.sheets['Validation']
        tempo = 3
        header_row = 2
        for i in experiments:
            worksheet.merge_range(header_row,tempo,header_row,tempo + 2,data= f'Step {i + 1}', cell_format = merge_format)
            tempo += 5
        print('Excel report generated: ', time)

def createLogFile(name, path):
    if name + '_log.csv' not in os.listdir(path):
        file = open(path + '/' + name +  '_log.csv', 'a')
        file.write('k-fold,ACC,PRE,MSE,REC,AUC,F1,MCC\n')
    else:
        file = open(path + '/' + name +  '_log.csv', 'a')
    return file

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
        file = createLogFile(name, path)
        if len(bestSubset) == 0:
            bestSubset = [1]
            print('>> Detectado caso 0')
        if runPipelineClassifiers == True:
            bestClassifier = classifiers.pipeline_classifiers(bestSubset, data, labels, 1, date_today, time, file, name)
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
            bestClassifier = classifiers.pipeline_classifiers(i[0], data, labels, times, date_today, time, file, i[2])
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
        plt.plot(x, historic, label = i[2] + ' Epochs: ' + str(i[3]))
    plt.xlabel('Epochs')
    plt.ylabel('Objective function value')
    plt.legend()
    plt.savefig(path + '/fig_'+time)
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
                local_data = classifiers.generateSubBM(subset, data)
                indexes = subset
            if local_data.shape[1] > 10:
                if i == 'SA':
                    dic_metas[key] = MetaheuristicStateWrapper(metaheuristic.SimulatedAnnealing(experiment_index, local_data, labels, 40, generations, local_data.shape[1], 0, 0.8, ofunctions.allObjectiveFunctionWeka, j, indexes, path_report)) 
                    
                if i == 'GA':
                    dic_metas[key] = MetaheuristicStateWrapper(metaheuristic.GeneticAlgorithm(experiment_index, local_data,labels,generations, population, local_data.shape[1], 0.20, 20, ofunctions.allObjectiveFunctionWeka, j, indexes, path_report)) 
                    
                if i == 'DE':
                    print('Loading new data for de: ', local_data, local_data.shape)
                    dic_metas[key] = MetaheuristicStateWrapper(metaheuristic.DifferentialEvolution(experiment_index, local_data,labels, population, local_data.shape[1], generations, 0.5, 0.45, ofunctions.allObjectiveFunctionWeka, j, indexes, path_report)) 
            else:
                print('>> The subset length is less than the minimum length required - ', key)
        experiment_index = experiment_index + 1
    return dic_metas



if __name__ == "__main__":
    print("--------------- Metaheuristic Collection  ---------------")
    results = []
    data_set = None
    subset = [28,  2, 34, 25,  9, 10, 32, 27, 36,  6, 13, 29,  1,  5, 38, 23, 37, 14, 15,  7, 26]
    data = None
    labels = None
    objective_functions_list = [ofunctions.mi_sklearn, ofunctions.mi_sklearn_2, ofunctions.mi_prop, ofunctions.svmFitnessValue, ofunctions.knnFitnessValue]
    while True:
        print('Menu principal: \n1. Establecer dataset\n2. Procesar dataset\n3. Escojer algoritmos metaheuristicos\n4. Ejecutar algoritmos seleccionados\n5. Ejecutar modo wrapper\n6. Cargar y correr configuración\n7. Ejecutar pipeline de clasificadores a tabla de resumen\n8. Correr paso desde archivo de resumen\n0. Salir')
        main_option = int(input('>> Escoja una opción del menu principal: '))
        if main_option == 1:
            while True:
                print('Se encontraron los siguientes datasets (.csv): ')
                data_set = os.listdir('data')
                for i in range(len(data_set)):
                    print(str(i + 1) + '. ' + data_set[i])
                print('0. Salir')
                option = int(input('>> Seleccione un dataset o 0 si desea salir: '))
                if option == 0:
                    break
                else:
                    data_set = data_set[option - 1]
                data, labels = processDataSet(data_set)
                # data, labels = processSubset(subset, data_set)
                print(f'>> Se selecciono el dataset ({option}) de ', data.shape)
        elif main_option == 2:
            while True:
                print('Escoja un dataset para normalizar ')
        elif main_option == 3:
            print("---------------------- Algoritmos metaheuristicos --------------------\n1. Gray Wolf Feature Selection\n2. Binary Ant Colony\n3. Genetic Algorithm\n4. Particle Swarm Algorithm\n5. Differential Evolution\n6. Simulated Annealing\n0. Salir")
            option = int(input('>> Escoja el numero que identifica al algoritmo: '))
            mh_array = []
            while True:
                if option == 1:
                    printAlgorithm('Gray Wolf Algorithm')
                    gen = setParameters('Gray Wolf Algorithm')
                    of_index = selectObjectiveFunction()
                    mh_array.append(metaheuristic.GrayWolfAlgorithm(data, labels, 12, gen, objective_functions_list[of_index - 1]))
                elif option == 2:
                    printAlgorithm('Binary Ant Colony')
                    gen = setParameters('Binary Ant Colony')
                    of_index = selectObjectiveFunction()
                    mh_array.append(metaheuristic.AntColony(data, labels, gen, 1, 0.90, 1.5, 0.2, 0.1, 1, 0.1, objective_functions_list[of_index - 1]))
                elif option == 3: 
                    printAlgorithm('Genetic Algorithm')
                    gen = setParameters('Genetic Algorithm')
                    of_index = selectObjectiveFunction()
                    mh_array.append(metaheuristic.GeneticAlgorithm(data, labels, gen, 80,data.shape[1], 2, 0.8,  0.20, 15, objective_functions_list[of_index - 1]))
                elif option == 4:
                    printAlgorithm('Particle Swarm Algorithm')
                    gen = setParameters('Particle Swarm Algorithm')
                    of_index = selectObjectiveFunction()
                    problem = metaheuristic.PSOProblem(data, labels,objective_functions_list[of_index - 1])
                    task = metaheuristic.Task(problem, max_iters = gen, optimization_type= OptimizationType.MAXIMIZATION)
                    pso_algorithm = metaheuristic.ParticleSwarmOptimization(data.shape[1], 1234)
                    mh_array.append(metaheuristic.PSO(task, pso_algorithm, problem, objective_functions_list[of_index - 1]))
                elif option == 5:
                    printAlgorithm('Differential Evolution')
                    gen = setParameters('Differential Evolution')
                    of_index = selectObjectiveFunction()
                    de = metaheuristic.DifferentialEvolution(data, labels, 20, data.shape[1], gen, 0.5, 0.45, objective_functions_list[of_index - 1])
                    mh_array.append(de)
                elif option == 6:
                    printAlgorithm('Simulated Annealing')
                    gen = setParameters('Simulated Annealing')
                    of_index = selectObjectiveFunction()
                    sa = metaheuristic.SimulatedAnnealing(data, labels, 20, gen, data.shape[1], 1, 0.95, objective_functions_list[of_index - 1])
                    mh_array.append(sa)
                elif option == 0:
                    print('** Volviendo al menu principal **')
                    break
                print("---------------------- Algoritmos metaheuristicos --------------------\n1. Gray Wolf Feature Selection\n2. Binary Ant Colony\n3. Genetic Algorithm\n4. Particle Swarm Algorithm\n5. Differential Evolution\n6. Simulated Annealing\n0. Salir")
                option = int(input('>> Escoja una opción del menú: '))
        elif main_option == 4:
            results, pathExp, historic, historicBest = runExperiment(mh_array, data, labels)
            option = runNewStep()
            historicAccumaluted = [historicBest]
            if option == 1:
                print('Historic out: ', historic, ' Historic best: ', historicBest)
                #Index 9 is for auc mean
                bestIndex = historicBest[:, 9].astype(float).argsort()
                print('Best index: ', bestIndex)
                bestIndex = bestIndex[-1]
                bestMhSubset = historic[bestIndex][0]
                print('Best: ', bestMhSubset)
                value = askAgain()
                # np.savetxt(pathExp, results, delimiter= ',', fmt='%s')
                if value == 0 or value == 2:
                    generateResume(historicAccumaluted)
                    break
                else:
                    while True:
                        #Index 9 is for auc mean
                        bestIndex = historicBest[:, 9].astype(float).argsort()
                        print('Best index: ', bestIndex)
                        bestIndex = bestIndex[1]
                        bestMhSubset = historic[bestIndex][0]
                        print('Best: ', bestMhSubset)
                        structData = classifiers.generateSubBM(bestMhSubset, data)
                        for meta in mh_array:
                            meta.setDataSet(structData, labels)
                            meta.initialize()
                        results, pathExp, historic, historicBest = runExperiment(mh_array, structData, labels)
                        value = askAgain()
                        # np.savetxt(pathExp, results, delimiter= ',', fmt='%s')
                        historicAccumaluted.append(historicBest)
                        if value == 0 or value == 2:
                            break
                        else:
                            print('>> Ejecutando de nuevo el experimento')
            elif option == 2:
                while True:
                    reStructured = []
                    for i in historic:
                        print('i: ', i)
                        if len(i[0]) > 1:
                            structData = classifiers.generateSubBM(i[0], data)
                            obj = filter(lambda element: element.name == i[2],mh_array)
                            obj = list(obj)[0]
                            print(f'>> Iniciando nuevamente {obj.name}')
                            obj.setDataSet(structData, labels)
                            obj.initialize()
                            reStructured.append(obj)
                    historicBestOrder, subSetResultHistoric, historicToGetMax = runExperimentInOrder(reStructured, data, labels)
                    historic = subSetResultHistoric
                    historicAccumaluted.append(historicToGetMax)
                    optionOrder = runInOrder()
                    if optionOrder == 0 or optionOrder == 2:
                        break
                    else:
                        print('>> Ejecutando de nuevo el algoritmo')

            generateResume(historicAccumaluted)
        elif main_option == 5: 
            results, _, historic, historicBest = runExperiment(mh_array, data, labels, runPipelineClassifiers=False)
            option = runNewStep()
            historicAccumaluted = [historicBest]
            if option == 1:
                print('Historic out: ', historic, ' Historic best: ', historicBest)
                #Index 0 is for auc mean
                bestIndex = historicBest[:, 0].astype(float).argsort()
                print('Best index: ', bestIndex)
                bestIndex = bestIndex[-1]
                bestMhSubset = historic[bestIndex][0]
                print('Best: ', bestMhSubset)
                value = askAgain()
                # np.savetxt(pathExp, results, delimiter= ',', fmt='%s')
                if value == 0 or value == 2:
                    generateResumeWithoutPipeline(historicAccumaluted)
                    break
                else:
                    while True:
                        #Index 0 is for auc mean
                        bestIndex = historicBest[:, 0].astype(float).argsort()
                        print('Best index: ', bestIndex)
                        bestIndex = bestIndex[1]
                        bestMhSubset = historic[bestIndex][0]
                        print('Best: ', bestMhSubset)
                        structData = classifiers.generateSubBM(bestMhSubset, data)
                        for meta in mh_array:
                            meta.setDataSet(structData, labels)
                            meta.initialize()
                        results, _, historic, historicBest = runExperiment(mh_array, structData, labels)
                        value = askAgain()
                        # np.savetxt(pathExp, results, delimiter= ',', fmt='%s')
                        historicAccumaluted.append(historicBest)
                        if value == 0 or value == 2:
                            break
                        else:
                            print('>> Ejecutando de nuevo el experimento')
            elif option == 2:
                while True:
                    reStructured = []
                    for i in historic:
                        print('i: ', i)
                        if len(i[0]) > 1:
                            structData = classifiers.generateSubBM(i[0], data)
                            obj = filter(lambda element: element.name == i[2],mh_array)
                            obj = list(obj)[0]
                            print(f'>> Iniciando nuevamente {obj.name}')
                            obj.setDataSet(structData, labels)
                            obj.initialize()
                            reStructured.append(obj)
                    historicBestOrder, subSetResultHistoric, historicToGetMax = runExperimentInOrder(reStructured, data, labels, runPipelineClassifiers=False)
                    historic = subSetResultHistoric
                    historicAccumaluted.append(historicToGetMax)
                    optionOrder = runInOrder()
                    if optionOrder == 0 or optionOrder == 2:
                        break
                    else:
                        print('>> Ejecutando de nuevo el algoritmo')

            generateResumeWithoutPipeline(historicAccumaluted)
        elif main_option == 6:
            # OBJECTIVE_FUNCTIONS = 'gain_ratio', 'relieff','information_gain', 'chi2', 'sim_uncertainty', 'mi', 'relieff'
            OBJECTIVE_FUNCTIONS = ['relieff','sim_uncertainty']
            path_config = 'configs'
            config_list = os.listdir('configs')
            option = 0
            dict_config = {}
            with open(os.path.join(path_config, config_list[option])) as f:
                for line in f:
                    key, value = line.strip().split(':')
                    value = value.strip()
                    if key == 'METAHEURISTICS':
                        value = value.split(',')
                    dict_config[key] = value

            # print('LINES: ', lines)
            STEPS_EXPERIMENT = int(dict_config['STEPS_EXPERIMENT'])
            START_GEN = int(dict_config['START_GENERATIONS'])
            END_GEN = int(dict_config['END_GENERATIONS'])
            STEP_GEN = int(dict_config['STEP_GENERATIONS'])
            METAS_EXPERIMENT = dict_config['METAHEURISTICS']
            R_K_TIMES = dict_config['REPEATED_CROSS_FOLD_VALIDATION']

            #Dataset
            data, labels = processDataSet('normal_rtf.csv')
            first_gen = START_GEN + STEP_GEN

            now = datetime.now()
            date_experiment = str(now.strftime("%d_%m_%Y"))
            hour_experiment = str(now.strftime("%H_%M_%S"))
            path = f'results/{date_experiment}/{hour_experiment}'
            path_historic = f'{path}/historic'
            if os.path.exists(path) == False:
                os.makedirs(path)
                os.makedirs(path_historic)
  
            experiment = Experiment(STEPS_EXPERIMENT, len(METAS_EXPERIMENT))
            columns=['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'ACC_MEAN', 'ACC_DESV', 'PRE_MEAN', 'PRE_DESV', 'MSE_MEAN', 'MSE_DESV', 'REC_MEAN', 'REC_DESV', 'AUC_MEAN', 'AUC_DESV', 'F1_MEAN', 'F1_DESV', 'MCC_MEAN','MCC_DESV']
            results_df = pd.DataFrame(columns=columns)
            mh_dict = getMetasByObjectiveFunction(data,labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, first_gen, path_report=path_historic)
            
            for i in range(STEPS_EXPERIMENT):
                result_metas_by_generation = {}
                name_step = 'STEP ' + str(i)
                print(f">>>> Starting step experiment: {i}/{STEPS_EXPERIMENT}")
                #Here mh_dict is generated with all data
                keys_dict = list(mh_dict.keys())
                tempo = len(keys_dict)
                n_run = 1
                print('FIRST GEN: ', first_gen, ' END GEN: ', END_GEN, ' STEP GEN: ', STEP_GEN, ' RUN: ', n_run)
                for gen in range(first_gen, END_GEN + 1, STEP_GEN):
                    print(f">>> Starting epoch experiment: {gen}/{END_GEN}")
                    ctx = multiprocessing.get_context('spawn')
                    results = []
                    
                    print('>> TOTAL MH EVALUATIONS: ', len(keys_dict))
                    print(keys_dict)
                    #First: assign an executor for 6 executors
                    futures = []
                    futures_wrapper = []
                    process_assigned = 0
                    total_process = len(keys_dict)
                    for key_index in range(0, NUMBER_EXECUTORS):
                        if key_index == len(keys_dict):
                            break
                        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                        meta = mh_dict[keys_dict[key_index]].metaheuristic
                        mh_dict[keys_dict[key_index]].setProcessAssigned(True)
                        future = executor.submit(meta.runProcess)
                        futures.append(future)
                        futures_wrapper.append(ExecutorWrapper(executor, future))
                        process_assigned += 1                    
                    results_futures = []
                    completed_all = False
                    while experiment.checkMetaheuristicStates(mh_dict) == False:
                        done, pending = concurrent.futures.wait(futures, timeout=None, return_when='FIRST_COMPLETED')
                        for future in done:
                            params = future.result()
                            index_future = futures.index(future)
                            futures.pop(index_future)
                            futures_wrapper[index_future].stopExecutor()
                            futures_wrapper.pop(index_future)
                            key_index = params['name'] + '_' + params['of_name']
                            if params['reload_weka_jvm'] == True:
                                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                                mh_dict[key_index].metaheuristic.setParams(params)
                                future = executor.submit(mh_dict[key_index].metaheuristic.runProcess)
                                futures.append(future)
                                time.sleep(15)
                                futures_wrapper.append(ExecutorWrapper(executor, future))
                            else:
                                results_futures.append(params)
                                mh_dict[key_index].setDone(True)
                            gc.collect()
                        #Check remaining metas
                        executors_free = NUMBER_EXECUTORS - len(futures)
                        print(f'> Step {i}-({gen}/{END_GEN}): Futures availables: {executors_free}/{NUMBER_EXECUTORS}')
                        print(f'> Step {i}-({gen}/{END_GEN}): Process assigned: {process_assigned}/{total_process}')
                        if process_assigned == (total_process):
                            completed_all = True
                        if completed_all == False and executors_free > 0:
                            remaining_mh = []
                            for mh in mh_dict:
                                if mh_dict[mh].processAssigned() == False:
                                    remaining_mh.append(mh_dict[mh])
                            print(f'> Step {i}-{gen}/{END_GEN}: Remaning mhs evaluation: {total_process - len(remaining_mh)}/{total_process}')
                            for free, mh in zip(list(range(executors_free)), remaining_mh):
                                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                                meta = mh.metaheuristic
                                mh.setProcessAssigned(True)
                                future = executor.submit(meta.runProcess)
                                time.sleep(10)
                                futures.append(future)
                                futures_wrapper.append(ExecutorWrapper(executor, future))
                                process_assigned += 1
                            print(f'> Step {i}-{gen}/{END_GEN}: Updated mhs evaluation: {total_process - len(remaining_mh)}/{total_process}')
                        # time.sleep(5)
                    print(f'>> Got {len(results_futures)} results')
                    for params in results_futures:
                        experiment.addDataByMeta(params, i)
                    print(f'>> Finish gen: ', gen)
                    n_run += 1
                    for key in mh_dict:
                        mh_dict[key].metaheuristic.set_new_iterations(n_run * STEP_GEN)
                        mh_dict[key].setDone(False)
                        mh_dict[key].setProcessAssigned(False)
                    print(f">>> Finish Epoch experiment: {gen}/{END_GEN}")
                total_historic = experiment.getHistoricByStep(i)
                total_historic.to_csv(f'{path}/total_history_{i + 1}_report.csv')
                historic_by_step, historic_by_step_df = experiment.getBestByObjectiveFunction(i, get_df=True)
                historic_by_step_df.to_csv(f'{path}/step_{i + 1}_report.csv', index = False)
                mh_dict = getMetasByObjectiveFunction(data,labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, first_gen, historic_by_step, path_report=path_historic)
            # last_data_entry = experiment.getLastDataEntry()
            # columns=['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'ACC MEAN', 'ACC DESV', 'PRE MEAN', 'PRE DESV', 'MSE MEAN', 'MSE DESV', 'REC MEAN', 'REC DESV', 'AUC MEAN', 'AUC DESV', 'F1 MEAN', 'F1 DESV', 'MCC MEAN','MCC DESV'] 
            # results_df = pd.DataFrame(columns=columns)
            # for data_entry in last_data_entry:
            #     best_subset = ast.literal_eval(data_entry[2])
            #     params = {
            #         'model': data_entry[0],
            #         'generation': data_entry[1],
            #         'settings': data_entry[4],
            #         'of_name': data_entry[3],
            #         'best_subset': data_entry[2],
            #         'length': len(best_subset)
            #     }
            #     results_classifiers = classifiers.pipeline_updated(best_subset, data, labels, R_K_TIMES, params, columns)
            #     results_df =  pd.concat([results_df, results_classifiers], ignore_index=True)
            # results_df.to_csv(f'{path}/classifiers_report.csv', sep=',', index = False)
        elif main_option == 7:
            R_K_TIMES = 3
            columns=['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'ACC_MEAN', 'ACC_DESV', 'PRE_MEAN', 'PRE_DESV', 'MSE_MEAN', 'MSE_DESV', 'REC_MEAN', 'REC_DESV', 'AUC_MEAN', 'AUC_DESV', 'F1_MEAN', 'F1_DESV', 'MCC_MEAN','MCC_DESV', 'SPEC_MEAN', 'SPEC_DESV']
            columns_matrix =['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'tn_mean', 'fp_mean',
                             'fn_mean', 'tp_mean', 'tn_desv', 'fp_desv', 'fn_desv', 'tp_desv', ]
            results_df = pd.DataFrame(columns=columns)
            matrix_df = pd.DataFrame(columns = columns_matrix)
            STEP = '1'
            resume_table = pd.read_csv(f'resumes/minimum_length_resume_class_step_2_report_train_normal_rtf.csv', sep = ',')
            resume_table_np = resume_table.to_numpy()
            NAME_FILE = 'train_normal_rtf'
            data, labels = processDataSet(f'{NAME_FILE}.csv')
            date_now = datetime.now()
            date_str = date_now.strftime('%Y_%m_%d')
            time_str = date_now.strftime('%H_%M_%S')
            path = 'classifiers_results/' + NAME_FILE + '/' + date_str + '/' + time_str
            full_history_name_file = 'min_len_step_2_full_history.csv'
            if os.path.exists(path) == False:
                os.makedirs(path)

            with open(f'{path}/{full_history_name_file}', 'w') as file:
                file.write(f'Meta,OF,Generation,Length,Settings,Classifier,Fold,AUC,Accuracy,Precision,MSE,Recall,F1,MCC\n')

            for data_entry in resume_table_np:
                print(f'>> Starting with {data_entry[0]}-{data_entry[1]}')
                best_subset = ast.literal_eval(data_entry[3])
                params = {
                    'model': data_entry[0],
                    'generation': data_entry[2],
                    'settings': data_entry[5],
                    'of_name': data_entry[1],
                    'best_subset': data_entry[3],
                    'length': len(best_subset),
                    'root_path': path
                }

                weights_path = f'{path}/{data_entry[0]}/weights_{data_entry[4]}'
                if os.path.exists(weights_path) == False:
                    os.makedirs(weights_path)
                results_classifiers, matrix_results = classifiers.mlp_pipeline(best_subset, data, labels, R_K_TIMES, params, columns, columns_matrix ,f'{path}/{full_history_name_file}', weights_path)
                results_df =  pd.concat([results_df, results_classifiers], ignore_index=True)
                matrix_df = pd.concat([matrix_df, matrix_results], ignore_index = True)
            results_df.to_csv(f'{path}/class_step_{STEP}_report_{NAME_FILE}.csv', sep=',', index = False)
            matrix_df.to_csv(f'{path}/matrix_step_{STEP}_report_{NAME_FILE}.csv', sep=',', index = False)
        elif main_option == 8:
            #Run from step file resume
            #['gain_ratio', 'information_gain', 'chi2', 'sim_uncertainty', 'mi', 'relieff']
            OBJECTIVE_FUNCTIONS = ['gain_ratio', 'information_gain', 'chi2', 'sim_uncertainty', 'mi', 'relieff']
            path_config = 'configs'
            config_list = os.listdir('configs')
            option = 0
            dict_config = {}
            with open(os.path.join(path_config, config_list[option])) as f:
                for line in f:
                    key, value = line.strip().split(':')
                    value = value.strip()
                    if key == 'METAHEURISTICS':
                        value = value.split(',')
                    dict_config[key] = value

            # print('LINES: ', lines)
            STEPS_EXPERIMENT = int(dict_config['STEPS_EXPERIMENT'])
            START_GEN = int(dict_config['START_GENERATIONS'])
            END_GEN = int(dict_config['END_GENERATIONS'])
            STEP_GEN = int(dict_config['STEP_GENERATIONS'])
            METAS_EXPERIMENT = dict_config['METAHEURISTICS']
            R_K_TIMES = dict_config['REPEATED_CROSS_FOLD_VALIDATION']

            #Dataset
            data, labels = processDataSet('normal_rtf.csv')
            first_gen = START_GEN + STEP_GEN

            now = datetime.now()
            date_experiment = str(now.strftime("%d_%m_%Y"))
            hour_experiment = str(now.strftime("%H_%M_%S"))
            path = f'results/{date_experiment}/{hour_experiment}'
            path_historic = f'{path}/historic'
            if os.path.exists(path) == False:
                os.makedirs(path)
                os.makedirs(path_historic)
          
            experiment = Experiment(STEPS_EXPERIMENT, len(METAS_EXPERIMENT))
            columns=['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'ACC MEAN', 'ACC DESV', 'PRE MEAN', 'PRE DESV', 'MSE MEAN', 'MSE DESV', 'REC MEAN', 'REC DESV', 'AUC MEAN', 'AUC DESV', 'F1 MEAN', 'F1 DESV', 'MCC MEAN','MCC DESV']
            results_df = pd.DataFrame(columns=columns)
            
            #Read the best subsets of step table resume
            step_resume = pd.read_csv('resumes/sa_min_len_step_2_report.csv', sep = ',')
            step_resume_np = step_resume.to_numpy()
            mh_dict = getMetasByObjectiveFunction(data,labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, first_gen, step_resume_np,path_report=path_historic)


            for i in range(STEPS_EXPERIMENT):
                result_metas_by_generation = {}
                name_step = 'STEP ' + str(i)
                print(f">>>> Starting step experiment: {i}/{STEPS_EXPERIMENT}")
                #Here mh_dict is generated with all data
                keys_dict = list(mh_dict.keys())
                tempo = len(keys_dict)
                n_run = 1
                print('FIRST GEN: ', first_gen, ' END GEN: ', END_GEN, ' STEP GEN: ', STEP_GEN, ' RUN: ', n_run)
                for gen in range(first_gen, END_GEN + 1, STEP_GEN):
                    print(f">>> Starting epoch experiment: {gen}/{END_GEN}")
                    ctx = multiprocessing.get_context('spawn')
                    results = []
                    
                    print('>> TOTAL MH EVALUATIONS: ', len(keys_dict))
                    print(keys_dict)
                    #First: assign an executor for 6 executors
                    futures = []
                    futures_wrapper = []
                    process_assigned = 0
                    total_process = len(keys_dict)
                    for key_index in range(0, NUMBER_EXECUTORS):
                        if key_index == len(keys_dict):
                            break
                        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                        meta = mh_dict[keys_dict[key_index]].metaheuristic
                        mh_dict[keys_dict[key_index]].setProcessAssigned(True)
                        future = executor.submit(meta.runProcess)
                        futures.append(future)
                        futures_wrapper.append(ExecutorWrapper(executor, future))
                        process_assigned += 1                    
                    results_futures = []
                    completed_all = False
                    while experiment.checkMetaheuristicStates(mh_dict) == False:
                        done, pending = concurrent.futures.wait(futures, timeout=None, return_when='FIRST_COMPLETED')
                        for future in done:
                            params = future.result()
                            index_future = futures.index(future)
                            futures.pop(index_future)
                            futures_wrapper[index_future].stopExecutor()
                            futures_wrapper.pop(index_future)
                            key_index = params['name'] + '_' + params['of_name']
                            if params['reload_weka_jvm'] == True:
                                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                                mh_dict[key_index].metaheuristic.setParams(params)
                                future = executor.submit(mh_dict[key_index].metaheuristic.runProcess)
                                futures.append(future)
                                time.sleep(15)
                                futures_wrapper.append(ExecutorWrapper(executor, future))
                            else:
                                results_futures.append(params)
                                mh_dict[key_index].setDone(True)
                            gc.collect()
                        #Check remaining metas
                        executors_free = NUMBER_EXECUTORS - len(futures)
                        print(f'> Step {i}-({gen}/{END_GEN}): Futures availables: {executors_free}/{NUMBER_EXECUTORS}')
                        print(f'> Step {i}-({gen}/{END_GEN}): Process assigned: {process_assigned}/{total_process}')
                        if process_assigned == (total_process):
                            completed_all = True
                        if completed_all == False and executors_free > 0:
                            remaining_mh = []
                            for mh in mh_dict:
                                if mh_dict[mh].processAssigned() == False:
                                    remaining_mh.append(mh_dict[mh])
                            print(f'> Step {i}-({gen}/{END_GEN}): Remaning mhs evaluation: {total_process - len(remaining_mh)}/{total_process}')
                            for free, mh in zip(list(range(executors_free)), remaining_mh):
                                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                                meta = mh.metaheuristic
                                mh.setProcessAssigned(True)
                                future = executor.submit(meta.runProcess)
                                time.sleep(10)
                                futures.append(future)
                                futures_wrapper.append(ExecutorWrapper(executor, future))
                                process_assigned += 1
                            print(f'> Step {i}-({gen}/{END_GEN}): Updated mhs evaluation: {total_process - len(remaining_mh)}/{total_process}')
                        # time.sleep(5)
                    print(f'>> Got {len(results_futures)} results')
                    for params in results_futures:
                        experiment.addDataByMeta(params, i)
                    print(f'>> Finish gen: ', gen)
                    n_run += 1
                    for key in mh_dict:
                        mh_dict[key].metaheuristic.set_new_iterations(n_run * STEP_GEN)
                        mh_dict[key].setDone(False)
                        mh_dict[key].setProcessAssigned(False)
                    print(f">>> Finish Epoch experiment: {gen}/{END_GEN}")
                total_historic = experiment.getHistoricByStep(i)
                total_historic.to_csv(f'{path}/total_history_{i + 1}_report.csv')
                historic_by_step, historic_by_step_df = experiment.getBestByObjectiveFunction(i, get_df=True)
                historic_by_step_df.to_csv(f'{path}/step_{i + 1}_report.csv', index=False)
                mh_dict = getMetasByObjectiveFunction(data,labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, first_gen, historic_by_step, path_report=path_historic)

        elif main_option == 0:
            break