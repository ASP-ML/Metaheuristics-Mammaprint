import os

MAX_ITERATIONS_JVM_WEKA = 50
SET_MIN_LENGTH = True
MIN_LENGTH_SUBSET = 10

def writterHistoric(path: str, data: dict):
    if os.path.exists(path) == False:
        with open(path, 'w') as file:
            file.write('Gen,Epoch,Subset,Merit\n')
    else:
        with open(path, 'a') as file:
            file.write(f'{data["gen"]},{data["epoch"]},{data["subset"]},{data["merit"]}\n')

class Metaheuristic:
    def __init__(self):
        self.historic = []
        self.bestScoreValue = 0
        self.bestSubset = []
    
    def getHistoric(self):
        return self.historic
    
    def getBestSubset(self):
        return self.bestSubset
    
    def getTotalFeatures(self):
        return len(list(self.bestSubset))
