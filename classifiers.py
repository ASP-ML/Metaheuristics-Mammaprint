import math
import numpy as np
import pandas as pd
import os
import csv
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import tree
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import random

stratifiedKFold = RepeatedStratifiedKFold(
    n_splits=10, n_repeats=3, random_state=45)
STEP_EXP_FOLD = random.randint(1000, 9999)
tf.keras.utils.set_random_seed(26)

def model_callbacks(filepath):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    verbose=0,
                                                    monitor='val_auc',
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto')
    return checkpoint


def mcc_metric(y_true, y_pred):
    y_pred_rounded = tf.keras.backend.round(y_pred)
    y_true_float = tf.keras.backend.cast(y_true, tf.float32)
    y_pred_rounded_float = tf.keras.backend.cast(y_pred_rounded, tf.float32)

    tp = tf.keras.backend.sum(y_true_float * y_pred_rounded_float)
    tn = tf.keras.backend.sum(
        (1.0 - y_true_float) * (1.0 - y_pred_rounded_float))
    fp = tf.keras.backend.sum((1.0 - y_true_float) * y_pred_rounded_float)
    fn = tf.keras.backend.sum(y_true_float * (1.0 - y_pred_rounded_float))

    numerator = (tp * tn - fp * fn)
    denominator = tf.keras.backend.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + tf.keras.backend.epsilon()

    mcc = numerator / denominator
    return mcc

def specificity_metric(y_true, y_pred):
    y_pred_rounded = tf.keras.backend.round(y_pred)
    y_true_float = tf.keras.backend.cast(y_true, tf.float32)
    y_pred_rounded_float = tf.keras.backend.cast(y_pred_rounded, tf.float32)

    tn = tf.keras.backend.sum((1.0 - y_true_float) * (1.0 - y_pred_rounded_float))
    fp = tf.keras.backend.sum((1.0 - y_true_float) * y_pred_rounded_float)
    
    specificity = tn / (tn + fp + tf.keras.backend.epsilon())
    return specificity

def f1_score_metric(y_true, y_pred):
    y_pred_rounded = tf.keras.backend.round(y_pred)
    y_true_float = tf.keras.backend.cast(y_true, tf.float32)
    y_pred_rounded_float = tf.keras.backend.cast(y_pred_rounded, tf.float32)

    tp = tf.keras.backend.sum(y_true_float * y_pred_rounded_float)
    fp = tf.keras.backend.sum((1.0 - y_true_float) * y_pred_rounded_float)
    fn = tf.keras.backend.sum(y_true_float * (1.0 - y_pred_rounded_float))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1_score = 2 * (precision * recall) / (precision +
                                           recall + tf.keras.backend.epsilon())
    return f1_score


def generateSubBM(features, BM):
    n, _ = BM.shape
    subBM = np.zeros((n, len(features)), dtype=float)
    for idx, value in enumerate(features):
        subBM[:, idx] = BM[:, value].reshape(n,)
    return subBM


def writeHistory(file, experiment_info, values):
    file.write(f'{experiment_info["model"]},{experiment_info["of_name"]},{experiment_info["generation"]},{experiment_info["length"]},{experiment_info["settings"]},{values["classifier"]},{values["fold"]},{values["auc"]},{values["accu"]},{values["pre"]},{values["mse"]},{values["recall"]},{values["f1"]},{values["mcc"]},{values["spec"]}\n')


def meanDev(scores, infoMh, name='Classifier undefined'):
    meta_name = infoMh['model']
    of_name = infoMh['of_name']
    generations = infoMh['generation']
    settings = infoMh['settings']
    best_subset = infoMh['best_subset']
    length = infoMh['length']
    metrics_array = np.array(
        [meta_name, of_name,  generations, best_subset, length, settings, name])
    confusion_array = np.array(
        [meta_name, of_name,  generations, best_subset, length, settings, name])

    for key in scores.keys():
        
        if key != 'confusion_matrix':
            mean = np.mean(np.array(scores[key]))
            desv = np.std(np.array(scores[key]))
            metrics_array = np.append(metrics_array, mean)
            metrics_array = np.append(metrics_array, desv)
        elif key == 'confusion_matrix':
            # Convert list of matrices to numpy array for easier computation
            matrices = np.array(scores[key])
            
            # Calculate mean matrix
            mean_matrix = np.mean(matrices, axis=0)
            confusion_array = np.append(confusion_array, mean_matrix.flatten())
            
            # Calculate standard deviation matrix
            std_matrix = np.std(matrices, axis=0)
            confusion_array = np.append(confusion_array, std_matrix.flatten())
            
            sum_matrix = np.sum(matrices, axis=0)
            confusion_array = np.append(confusion_array, sum_matrix.flatten())
    return metrics_array, confusion_array

def get_metrics(y_predict, y_probability, y_test):
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    mse = metrics.mean_squared_error(y_test, y_predict)
    #tn, fp, fn, tp
    confusion_matrix, auc, f1, mcc = metrics.confusion_matrix(y_test, y_predict), metrics.roc_auc_score(
        y_test,  y_probability[:, 1]), metrics.f1_score(y_test, y_predict), metrics.matthews_corrcoef(y_test, y_predict)
    return accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity

def testKnn(k, X, y, experiment_info, file=None, times=1, name='KNN'):
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}

    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn = KNeighborsClassifier(k)
        knn.fit(X_train, y_train)
        y_predict = knn.predict(X_test)
        y_probability = knn.predict_proba(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def testNaiveBayes(X, y, experiment_info, file=None, times=1, name='Naive bayes'):
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}
    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_predict = gnb.predict(X_test)
        y_probability = gnb.predict_proba(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def testSVC(kernelP, X, y, experiment_info, file=None, times=1, name='SVM'):
    
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}
    
    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(kernel=kernelP, probability=True)
        clf.fit(X_train, y_train)
        y_probability = clf.predict_proba(X_test)
        y_predict = clf.predict(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def testCartTree(X, y, experiment_info, file=None, times=1, name='CART'):
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}
    

    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_probability = clf.predict_proba(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def testRandomForest(X, y, n_estimators,  experiment_info, file=None, times=1, name='RF'):
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}
    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators, random_state=45)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_probability = clf.predict_proba(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def testMLP(features, X, y, epochs, experiment_info, file, history_file_path, weights_path, name='MLP'):


    scoresCustom = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}

    k_fold = 1
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = tf.keras.Sequential()
        il = len(features)
        h1_l = math.trunc((len(features) + 2) / 2)

        model.add(tf.keras.layers.Input(il))
        model.add(tf.keras.layers.Dense(
            units=(h1_l), activation='sigmoid'))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        sgd = tf.keras.optimizers.SGD(
            learning_rate= 0.03,
            momentum=0.2
        )
        model.compile(optimizer=sgd,
                        loss=tf.keras.losses.MeanSquaredError(), metrics=['AUC', 'Accuracy',  'Precision', 'Recall', 'MeanSquaredError', mcc_metric, f1_score_metric, specificity_metric])
        historiesBasePath = f'histories/{STEP_EXP_FOLD}_{experiment_info["model"]}_{experiment_info["of_name"]}_{experiment_info["length"]}/'

        if os.path.exists(historiesBasePath) == False:
            os.mkdir(historiesBasePath)
        MLP_filepath = weights_path + "/MLP_weights_best_epoch_{epoch:02d}" + f"_fold_{k_fold}"+"_val_auc_{val_auc:.2f}.hdf5"
        MLP_checkpoint = model_callbacks(MLP_filepath)
        history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data= (X_test, y_test),
                            verbose=False, callbacks=MLP_checkpoint)
        history_train = history.history
        

        history_CNN_DF = pd.DataFrame.from_dict(history.history)
        history_CNN_DF.to_csv(
            historiesBasePath + 'MLP_historyCSV_fold_' + str(k_fold) + '.csv')

        history_train['fold'] = [str(k_fold) for _ in range(epochs)]
        history_train['epoch'] = [i + 1 for i in range(epochs)]

        for epoch in range(epochs):
            loss_k = history_train['val_loss'][epoch]
            auc_k = history_train['val_auc'][epoch]
            acc_k = history_train['val_Accuracy'][epoch]
            pre_k = history_train['val_precision'][epoch]
            mse_k = history_train['val_mean_squared_error'][epoch]
            mcc_k = history_train['val_mcc_metric'][epoch]
            f1_k = history_train['val_f1_score_metric'][epoch]
            recall_k = history_train['val_recall'][epoch]
            specificity_k = history_train['val_specificity_metric'][epoch]
            model_meta = experiment_info["model"]
            of_name = experiment_info["of_name"]
            gen = experiment_info["generation"]
            length = experiment_info["length"]
            settings = experiment_info["settings"]
            classifier = f'MLP {epoch + 1}'
            file.write(f"{model_meta},{of_name},{gen},{length},{settings},{classifier},{k_fold},{auc_k},{acc_k},{pre_k},{mse_k},{recall_k},{f1_k},{mcc_k},{loss_k},{specificity_k}\n")
        with open(history_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(zip(*history_train.values()))
        y_proba = model.predict(X_test)
        y_predict = np.where(y_proba > 0.499999999, 1, 0)
        y_predict = np.squeeze(y_predict)
        accuracy, mse, precision, recall = metrics.accuracy_score(y_test, y_predict), metrics.mean_squared_error(
            y_test, y_predict), metrics.precision_score(y_test, y_predict), metrics.recall_score(y_test, y_predict)
        confusion_matrix, auc, f1, mcc = metrics.confusion_matrix(y_test, y_predict), metrics.roc_auc_score(
            y_test, y_proba), metrics.f1_score(y_test, y_predict), metrics.matthews_corrcoef(y_test, y_predict)
        scoresCustom['accuracy'].append(accuracy)
        scoresCustom['precision'].append(precision)
        scoresCustom['mse'].append(mse)
        scoresCustom['recall'].append(recall)
        scoresCustom['auc'].append(auc)
        scoresCustom['f1'].append(f1)
        scoresCustom['mcc'].append(mcc)
        scoresCustom['confusion_matrix'].append(confusion_matrix),
        scoresCustom['specificity'].append(specificity_metric(y_test, y_proba))
        k_fold += 1
    return scoresCustom

def testLDA(X, y, experiment_info, file=None, times=1, name='LDA'):
    
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}
    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_probability = clf.predict_proba(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def testQDA(X, y, experiment_info, file=None, times=1, name='QDA'):
   
    scores = {'accuracy': [], 'precision': [], 'mse': [], 'recall': [
    ], 'auc': [], 'f1': [], 'mcc': [], 'confusion_matrix': [], 'specificity': []}

    k_fold = 0
    for train_index, test_index in stratifiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_probability = clf.predict_proba(X_test)
        accuracy, precision, mse, recall, confusion_matrix, auc, f1, mcc, specificity = get_metrics(y_predict, y_probability, y_test)

        scores['accuracy'].append(accuracy)
        scores['precision'].append(precision)
        scores['mse'].append(mse)
        scores['recall'].append(recall)
        scores['auc'].append(auc)
        scores['f1'].append(f1)
        scores['mcc'].append(mcc)
        scores['confusion_matrix'].append(confusion_matrix)
        scores['specificity'].append(specificity)
        k_fold += 1
        values = {'classifier': name, 'fold': str(k_fold), 'auc': str(auc), 'accu': str(accuracy), 'pre': str(
            precision), 'mse': str(mse), 'recall': str(recall), 'f1': str(f1), 'mcc': str(mcc), 'spec': str(specificity)}
        writeHistory(file, experiment_info, values)
    return scores


def runKnn(data, labels, k_times, infoWh, file=None):
    print('>> Running KNN')
    kList = [3, 5, 7, 9, 11]
    knnResultMetrics = []
    knnMatrix = []
    for k in kList:
        scoresDict = testKnn(k, data, labels, infoWh,
                             file, name=f'Knn k = {k}')
        metricsArray, confusionArray = meanDev(
            scoresDict, infoWh, f'Knn k = {k}')
        knnResultMetrics.append(metricsArray)
        knnMatrix.append(confusionArray)
    return knnResultMetrics, knnMatrix


def runNaiveBaiyes(data, labels, k_times, infoWh, file=None):
    print('>> Running Naive Bayes')
    scores = testNaiveBayes(data, labels, infoWh, file)
    metricsArrayNaive, confusionNaive = meanDev(scores, infoWh, 'Naive Bayes')
    return [metricsArrayNaive], [confusionNaive]


def runCartTree(data, labels, k_times, infoWh, file=None):
    print('>> Running Cart Tree')
    scores = testCartTree(data, labels)
    metricsArrayTree, confusionTree = meanDev(scores, infoWh, 'Cart Tree')
    return [metricsArrayTree], [confusionTree]


def runSVC(data, labels, k_times, infoWh, file=None):
    print('>> Running SVC')
    kernelsList = ['poly']
    svcMetrics = []
    svcMatrix = []
    for k in kernelsList:
        scoresDict = testSVC(k, data, labels, infoWh, file, name='SVC ' + k)
        metricsArray, confusionArray = meanDev(
            scoresDict, infoWh, f'SVM Kernel = {k}')
        svcMetrics.append(metricsArray)
        svcMatrix.append(confusionArray)
    return svcMetrics, svcMatrix


def runMLP(features, data, labels, k_times, infoWh, weigths_path, file=None):
    mlpResultMetrics = []
    mlpMatrix = []

    for i in range(0,500, 50):
        # i = 500
        print('>> Running Feed forward backpropagation ' + str(i))
        name_history = f"{infoWh['root_path']}/{infoWh['model']}_{infoWh['of_name']}_{infoWh['length']}.csv"
        columns = ["loss", "auc", "Accuracy", "precision", "recall",
                "mean_squared_error", "mcc_metric", "f1_score_metric", "specificity_metric", "fold", "epoch"]
        with open(name_history, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
        scores = testMLP(features, data, labels, i, infoWh,
                        file, name_history, weigths_path, f'MLP - {i}')
        metricsArray, confusionArray = meanDev(scores, infoWh, f'MLP - {i}')
        mlpResultMetrics.append(metricsArray)
        mlpMatrix.append(confusionArray)
        return mlpResultMetrics, mlpMatrix


def runLDA(data, labels, k_times, infoWh, file=None):
    print('>> Running LDA')
    scores = testLDA(data, labels, infoWh, file)
    metricsArray, confusionArray = meanDev(scores, infoWh, 'LDA')
    return [metricsArray], [confusionArray]


def runQDA(data, labels, k_times, infoWh, file=None):
    print('>> Running QDA')
    scores = testQDA(data, labels, infoWh, file)
    metricsArray, confusionArray = meanDev(scores, infoWh, 'QDA')
    return [metricsArray], [confusionArray]


def runRandomForest(data, labels, k_times, infoWh, file=None):
    rfResultMetrics = []
    rfMatrix = []
    for i in range(1, 101):
        print(f">> Running Random forest with {i}")
        scores = testRandomForest(
            data, labels, i, infoWh, file, name='RF estimators: ' + str(i))
        metricsArray, confusionArray = meanDev(
            scores, infoWh, 'RF estimators: ' + str(i))
        rfResultMetrics.append(metricsArray)
        rfMatrix.append(confusionArray)
    return rfResultMetrics, rfMatrix


def addRowsDataFrame(df, matrix):
    for i in matrix:
        df.loc[len(df.index)] = i


def pipeline_classifiers(features, data, labels, ktimes, date_today, time, file, name_classifier='undefined', returnBestClassifier=True):
    results_df = pd.DataFrame(columns=['Classifier', 'ACC MEAN', 'ACC DESV', 'PRE MEAN', 'PRE DESV', 'MSE MEAN',
                              'MSE DESV', 'REC MEAN', 'REC DESV', 'AUC MEAN', 'AUC DESV', 'F1 MEAN', 'F1 DESV', 'MCC MEAN', 'MCC DESV'])
    #tn, fp, fn, tp
    matrix_df = pd.DataFrame(columns=['Classifier', 'tn mean', 'fp mean',
                             'fn mean', 'tp mean', 'tn desv', 'fp desv', 'fn desv', 'tp desv', 'tn sum', 'fp sum',
                             'fn sum', 'tp sum'])
    structured_data = generateSubBM(features, data)
    knn_results_metrics, knn_results_matrix = runKnn(
        structured_data, labels, ktimes, file)
    naive_results_metrics, naive_results_matrix = runNaiveBaiyes(
        structured_data, labels, ktimes, file)
    # tree_results_metrics, tree_results_matrix = runCartTree(structured_data, labels, ktimes, file)
    svc_results_metrics, svc_results_matrix = runSVC(
        structured_data, labels, ktimes, file)
    mlp_results_metrics, mlp_results_matrix = runMLP(
        features, structured_data, labels, ktimes, file)
    lda_results_metrics, lda_results_matrix = runLDA(
        structured_data, labels, ktimes, file)
    qda_results_metrics, qda_results_matrix = runQDA(
        structured_data, labels, ktimes, file)
    rf_results_metrics, rf_results_matrix = runRandomForest(
        structured_data, labels, ktimes, file)

    addRowsDataFrame(results_df, knn_results_metrics)
    addRowsDataFrame(results_df, naive_results_metrics)
    # addRowsDataFrame(results_df, tree_results_metrics)
    addRowsDataFrame(results_df, svc_results_metrics)
    addRowsDataFrame(results_df, mlp_results_metrics)
    addRowsDataFrame(results_df, lda_results_metrics)
    addRowsDataFrame(results_df, qda_results_metrics)
    addRowsDataFrame(results_df, rf_results_metrics)

    addRowsDataFrame(matrix_df, knn_results_matrix)
    addRowsDataFrame(matrix_df, naive_results_matrix)
    # addRowsDataFrame(matrix_df, tree_results_matrix)
    addRowsDataFrame(matrix_df, svc_results_matrix)
    addRowsDataFrame(matrix_df, mlp_results_matrix)
    addRowsDataFrame(matrix_df, lda_results_matrix)
    addRowsDataFrame(matrix_df, qda_results_matrix)
    addRowsDataFrame(matrix_df, rf_results_matrix)

    path = 'results/' + date_today + '/' + time
    if os.path.exists(path) == False:
        os.makedirs(path)
    results_df.to_csv(path + '/' + name_classifier +
                      '_results_' + time + '.csv', sep=',', index=False)
    matrix_df.to_csv(path + '/' + name_classifier +
                     '_matrix_' + time + '.csv', sep=',', index=False)
    print(f'>> {time} - Pipeline listo')
    #Returning the best value
    if returnBestClassifier:
        return results_df.loc[results_df['AUC MEAN'].astype(float).idxmax()].to_numpy()
    return None


def pipeline_updated(features, data, labels, ktimes, infoMh, result_columns, matrix_columns, full_history_name_file, weights_path):
    results_df = pd.DataFrame(columns=result_columns)
    matrix_df = pd.DataFrame(columns=matrix_columns)
    file = open(full_history_name_file, 'a')
    
    structured_data = generateSubBM(features, data)

    knn_results_metrics, knn_results_matrix = runKnn(structured_data, labels, ktimes, infoMh, file)
    naive_results_metrics, naive_results_matrix = runNaiveBaiyes(structured_data, labels, ktimes, infoMh, file)
    svc_results_metrics, svc_results_matrix = runSVC(structured_data, labels, ktimes, infoMh, file)
    # mlp_results_metrics, mlp_results_matrix = runMLP(
    #    features, structured_data, labels, ktimes, infoMh, weights_path, file)
    lda_results_metrics, lda_results_matrix = runLDA(structured_data, labels, ktimes, infoMh, file)
    # qda_results_metrics, qda_results_matrix = runQDA(structured_data, labels, ktimes, infoMh, file = file)
    # rf_results_metrics, rf_results_matrix = runRandomForest(structured_data, labels, ktimes, infoMh, file)
    file.close()
    addRowsDataFrame(results_df, knn_results_metrics)
    addRowsDataFrame(results_df, naive_results_metrics)
    addRowsDataFrame(results_df, svc_results_metrics)
    # addRowsDataFrame(results_df, mlp_results_metrics)
    addRowsDataFrame(results_df, lda_results_metrics)
    # addRowsDataFrame(results_df, qda_results_metrics)
    # addRowsDataFrame(results_df, rf_results_metrics)

    addRowsDataFrame(matrix_df, knn_results_matrix)
    addRowsDataFrame(matrix_df, naive_results_matrix)
    addRowsDataFrame(matrix_df, svc_results_matrix)
    # addRowsDataFrame(matrix_df, mlp_results_matrix)
    addRowsDataFrame(matrix_df, lda_results_matrix)
    # addRowsDataFrame(matrix_df, qda_results_matrix)
    # addRowsDataFrame(matrix_df, rf_results_matrix)

    return results_df, matrix_df


def mlp_pipeline(features, data, labels, ktimes, infoMh, result_columns, matrix_columns, full_history_name_file, weights_path):
    results_df = pd.DataFrame(columns=result_columns)
    matrix_df = pd.DataFrame(columns=matrix_columns)
    file = open(full_history_name_file, 'a')
    structured_data = generateSubBM(features, data)
    mlp_results_metrics, mlp_results_matrix = runMLP(
        features, structured_data, labels, ktimes, infoMh, weights_path, file)
    file.close()
    addRowsDataFrame(results_df, mlp_results_metrics)
    addRowsDataFrame(matrix_df, mlp_results_matrix)
    return results_df, matrix_df

def qda_pipeline(features, data, labels, ktimes, infoMh, result_columns, matrix_columns, full_history_name_file):
    results_df = pd.DataFrame(columns=result_columns)
    matrix_df = pd.DataFrame(columns=matrix_columns)
    file = open(full_history_name_file, 'a')
    structured_data = generateSubBM(features, data)
    qda_results_metrics, qda_results_matrix = runQDA(
        structured_data, labels, ktimes, infoMh, file)
    file.close()

    addRowsDataFrame(results_df, qda_results_metrics)
    addRowsDataFrame(matrix_df, qda_results_matrix)
    return results_df, matrix_df