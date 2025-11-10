import pandas as pd
import numpy as np
import shared_imports as si
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

def processSubset(features, data_name):
    data = pd.read_csv('data/' + data_name)
    labels = data['label'].to_numpy()
    data = data.drop(['label'], axis= 1).to_numpy()
    data = si.ofunctions.generateSubBM(features, data)
    return data, labels

def subsetByPath(features, path):
    data = pd.read_csv(path)
    labels = data['label'].to_numpy()
    data = data.drop(['label'], axis= 1).to_numpy()
    data = si.ofunctions.generateSubBM(features, data)
    return data, labels

def addRowsDataFrame(df, matrix):
    for i in matrix:
        df.loc[len(df.index)] = i
        
def createLogFile(name, path):
    if name + '_log.csv' not in si.os.listdir(path):
        file = open(path + '/' + name +  '_log.csv', 'a')
        file.write('k-fold,ACC,PRE,MSE,REC,AUC,F1,MCC\n')
    else:
        file = open(path + '/' + name +  '_log.csv', 'a')
    return file

def make_path_if_not_exists(path):
    if si.os.path.exists(path) == False:
        si.os.makedirs(path)

def make_experiment_dirs(date, time, auto_time_stamp=True):
    if auto_time_stamp:
        now = si.datetime.now()
        date = str(now.strftime("%d_%m_%Y"))
        time = str(now.strftime("%H_%M_%S"))
    else:
        date = 'default_date'
        time = 'default_time'
        
    
    base_path = f'results/{date}/{time}'
    make_path_if_not_exists(base_path)
    make_path_if_not_exists(f'{base_path}/historic')
    return base_path, f'{base_path}/historic'