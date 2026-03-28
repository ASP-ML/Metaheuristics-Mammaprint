import pandas as pd
from statistics import mean, stdev
import csv

def computeMeanDev(pathFile: str, pathFullHistory: str, pathMeanByFold: str):
    df = pd.read_csv(pathFile, sep=',')
    sortedDF = df.sort_values(['Meta','OF','Classifier', 'Fold'], ascending=[True, True, True, True])
    #sortedDF.to_csv('sorted.csv', sep=',', index=False)
    foldCol = sortedDF.loc[:, 'Fold']
    indexes = [i for i in range(foldCol.size)]
    #print('COL: ', foldCol)
    #print('INDEXES: ', indexes)
    columns = ["Meta","OF","Generation","Length","Settings","Classifier","AUC_MEAN","AUC_STD", "ACC_MEAN", "ACC_STD","PRE_MEAN","PRE_STD","MSE_MEAN","MSE_STD","REC_MEAN","REC_STD","F1_MEAN","F1_STD","MCC_MEAN", "MCC_STD", "SPEC_MEAN", "SPEC_STD"]

    columnsWithFoldMean = ["Meta","OF","Generation","Length","Settings","Classifier","Fold","F1_MEAN","AUC", "ACC", "PRE","MSE","REC","F1","MCC", "SPEC"]


    f1 = open(pathFullHistory, 'w')
    f = open(pathMeanByFold, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(columns)
    writer1 = csv.writer(f1, lineterminator='\n')
    writer1.writerow(columnsWithFoldMean)

    folds = 30 
    for index in range(0, len(indexes), 30):
        print('>>> Computing: ', index)
        train_accuracy, train_precision, train_recall, train_auc, train_f1, train_mcc, train_mse, train_spec, folds_list = list(), list(), list(), list(), list(),list(),list(), list(), list()
        for i in range(index, index + folds):
            train_accuracy.append(sortedDF.iloc[indexes[i]]['Accuracy'])
            train_precision.append(sortedDF.iloc[indexes[i]]['Precision'])
            train_mse.append(sortedDF.iloc[indexes[i]]['MSE'])
            train_recall.append(sortedDF.iloc[indexes[i]]['Recall'])
            train_f1.append(sortedDF.iloc[indexes[i]]['F1'])
            train_mcc.append(sortedDF.iloc[indexes[i]]['MCC'])
            train_auc.append(sortedDF.iloc[indexes[i]]['AUC'])
            train_spec.append(sortedDF.iloc[indexes[i]]['Specificity'])
            folds_list.append(sortedDF.iloc[indexes[i]]['Fold'])

        nameMeta = sortedDF.iloc[indexes[i]]['Meta']
        nameOF = sortedDF.iloc[indexes[i]]['OF']
        gen = sortedDF.iloc[indexes[i]]['Generation']
        length = sortedDF.iloc[indexes[i]]['Length']
        settings = sortedDF.iloc[indexes[i]]['Settings']
        classifier = sortedDF.iloc[indexes[i]]['Classifier']

        mean_accuracy = mean(train_accuracy)
        mean_precision = mean(train_precision)
        mean_mse = mean(train_mse)
        mean_recall = mean(train_recall)
        mean_f1 = mean(train_f1)
        mean_auc = mean(train_auc)
        mean_mcc = mean(train_mcc)
        mean_spec = mean(train_spec)

        std_accuracy = stdev(train_accuracy)
        std_precision = stdev(train_precision)
        std_mse = stdev(train_mse)
        std_recall = stdev(train_recall)
        std_f1 = stdev(train_f1)
        std_auc = stdev(train_auc)
        std_mcc = stdev(train_mcc)
        std_spec = stdev(train_spec)

        v = [nameMeta, nameOF, gen, length, settings, classifier, str(mean_auc), str(std_auc), str(mean_accuracy), str(std_accuracy), str(mean_precision), str(std_precision), str(mean_mse), str(std_mse), str(mean_recall), str(std_recall), str(mean_f1), str(std_f1), str(mean_mcc), str(std_mcc), str(mean_spec), str(std_spec)]

        writer.writerow(v)
        for j in range(len(folds_list)):
            v = [nameMeta, nameOF, gen, length, settings, classifier, str(folds_list[j]), str(mean_f1), str(train_auc[j]), str(train_accuracy[j]), str(train_precision[j]), str(train_mse[j]), str(train_recall[j]),  str(train_f1[j]), str(train_mcc[j]), str(train_spec[j])]
            writer1.writerow(v)

    f.close()
    f1.close()
if __name__ == "__main__":
    print('-------- Process classifiers --------')
    # baseDir = 'INTERSECTION_5'
    # pathFile = f'{baseDir}/min_len_step_3_full_history.csv'
    # pathMeanHistoryByFold = f'{baseDir}/meanFullHistoryByFold.csv'
    # pathMeanByFold = f'{baseDir}/meanByFold.csv'
    # computeMeanDev(pathFile, pathMeanHistoryByFold, pathMeanByFold)
    baseDir = 'results'
    pathFile = f'{baseDir}/lasso_full_history.csv'
    pathMeanHistoryByFold = f'{baseDir}/meanFullHistoryByFold.csv'
    pathMeanByFold = f'{baseDir}/meanByFold.csv'
    
    #Change the variable to choose a pivot
    computeMeanDev(pathFile, pathMeanHistoryByFold, pathMeanByFold)

