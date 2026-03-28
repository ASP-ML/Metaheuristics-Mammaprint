from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
import seaborn as sns
import ast

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# sns.set(rc = {'figure.figsize':(8,5)})
# sns.set(font_scale = 1.4)

function_map = {
    'gain_ratio': 'aGR',
    'mi': 'aMI',
    'sim_uncertainty': 'aSU',
    'chi2': 'aCHI2',
    'relieff': 'aReliefF',
}

kf = RepeatedStratifiedKFold(
n_splits=10, n_repeats=3, random_state=45)
plt.figure(figsize=(13, 10))
font = FontProperties(family='Verdana',  size=22)

def getClassifierName(name):
    config = ''
    parsed_name = ''
    type_classifier = ''
    if name.startswith('Knn'):
        config = int(name[len("Knn k = "):])
        parsed_name = f'KNN (k = {config})'
        type_classifier = 'knn'
    elif name.startswith('RF'):
        config = int(name[len("RF estimators: "):])
        parsed_name = f'RF (t = {config})'
        type_classifier = 'rf'
    elif name.startswith('SVM'):
        type_classifier = 'svm'
        parsed_name = 'SVM (Poly)'
    elif name.startswith('NB') or name.startswith('Naive'):
        type_classifier = 'nb'
    print('NAME: ', name)
    print(type_classifier)
    return config, parsed_name, type_classifier

def processDataSet(data_name):
    data = pd.read_csv('data/' + data_name)
    labels = data['label'].to_numpy(dtype=np.intp)
    data = data.drop(['label'], axis= 1).to_numpy()
    return data, labels

def getMetrics(Y_test, Y_predict, Y_probability):
    accuracy = metrics.accuracy_score(Y_test, Y_predict)
    mse = metrics.mean_squared_error(Y_test, Y_predict)
    precision = metrics.precision_score(Y_test, Y_predict)
    recall = metrics.recall_score(Y_test, Y_predict)
    auc = metrics.roc_auc_score(Y_test, Y_probability)
    f1 = metrics.f1_score(Y_test, Y_predict)
    mcc = metrics.matthews_corrcoef(Y_test, Y_predict)
    dict_metrics = {
        'acc': accuracy,
        'mse': mse,
        'pre': precision,
        'rec': recall,
        'auc': auc,
        'f1': f1,
        'mcc': mcc,
    }
    return dict_metrics

def generateSubBM(features, BM):
    n, _ = BM.shape
    subBM = np.zeros((n, len(features)), dtype=float)
    for idx, value in enumerate(features):
        subBM[:, idx] = BM[:, value].reshape(n,)
    return subBM

def auc_roc_rf(sub_X, labels, estimators):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold = 0
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Initialize and train the model (e.g., Logistic Regression)
        model = RandomForestClassifier(estimators, random_state=45)
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        # Interpolate the current TPR values to the common set of FPR values
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        fold += 1
    return tprs, mean_fpr

def auc_roc_svm(sub_X, labels):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold = 0
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Initialize and train the model (e.g., Logistic Regression)
        model = SVC(kernel='linear', probability=True, random_state=45)
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        # Interpolate the current TPR values to the common set of FPR values
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        fold += 1
    return tprs, mean_fpr

def auc_roc_nb(sub_X, labels):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold = 0
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Initialize and train the model (e.g., Logistic Regression)
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        # Interpolate the current TPR values to the common set of FPR values
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        fold += 1
    return tprs, mean_fpr

def auc_roc_knn(sub_X, labels, k):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    fold = 0
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = KNeighborsClassifier(k)
        model.fit(X_train, y_train)
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        fold += 1
    return tprs, mean_fpr


# --- PR-AUC data-collection functions ---

def pr_auc_rf(sub_X, labels, estimators):
    precisions = []
    mean_recall = np.linspace(0, 1, 100)
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = RandomForestClassifier(estimators, random_state=45)
        model.fit(X_train, y_train)
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
    return precisions, mean_recall

def pr_auc_svm(sub_X, labels):
    precisions = []
    mean_recall = np.linspace(0, 1, 100)
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = SVC(kernel='linear', probability=True, random_state=45)
        model.fit(X_train, y_train)
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
    return precisions, mean_recall

def pr_auc_nb(sub_X, labels):
    precisions = []
    mean_recall = np.linspace(0, 1, 100)
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
    return precisions, mean_recall

def pr_auc_knn(sub_X, labels, k):
    precisions = []
    mean_recall = np.linspace(0, 1, 100)
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = KNeighborsClassifier(k)
        model.fit(X_train, y_train)
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
    return precisions, mean_recall


def precision_vs_recall_curve_rf(sub_X, labels, estimators):
    y_real = []
    y_proba = []
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Initialize and train the model (e.g., Logistic Regression)
        model = RandomForestClassifier(estimators, random_state=45)
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_real.append(y_test)
        y_proba.append(y_pred_prob)
    return y_real, y_proba

def precision_vs_recall_curve_knn(sub_X, labels, k):
    y_real = []
    y_proba = []
    for train_index, test_index in kf.split(sub_X, labels):
        X_train, X_test = sub_X[train_index], sub_X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Initialize and train the model (e.g., Logistic Regression)
        model = KNeighborsClassifier(k)
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_real.append(y_test)
        y_proba.append(y_pred_prob)
    return y_real, y_proba

def plot_pre_vs_rec(subset, X, labels, classifier_obj, ax):
    sub_X = generateSubBM(subset, X)
    #sub_X_2 = generateSubBM([6, 9, 11, 16, 18, 19, 21, 24, 29, 30, 31, 32, 35, 38, 39, 42, 43, 50, 51, 57, 58, 63, 65, 67, 68], X)

    config = classifier_obj['config']
    type_classifier = classifier_obj['type_classifier']
    name_classifier = classifier_obj['name_classifier']
    label = classifier_obj['label']
    COLOR = classifier_obj['color']
    
    precision, recall = None, None
    if type_classifier == 'rf':
        y_real, y_proba = precision_vs_recall_curve_rf(sub_X, labels, config)
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
    elif type_classifier == 'knn':
        y_real, y_proba = precision_vs_recall_curve_knn(sub_X, labels, config)
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    # precision[0] = 0
    mAP = metrics.average_precision_score(y_real, y_proba)
    ax.plot(recall, precision, color=COLOR, label=f'{label} = {mAP:.3f})')

    #y_real_2, y_proba_2 = precision_vs_recall_curve(sub_X_2, labels, 100)
    #y_real_2 = np.concatenate(y_real_2)
    #y_proba_2 = np.concatenate(y_proba_2)
    #precision_2, recall_2, _ = precision_recall_curve(y_real_2, y_proba_2)
    # precision_2[0] = 0
    #mAP = metrics.average_precision_score(y_real_2, y_proba_2)
    #plt.plot(recall_2, precision_2, color='orange', label=f'RF (t = 80) - AP = {mAP:.3f})')
    
# classifier, name_classifier = 'undefined'
def plot_auc(subset, X, labels, classifier_obj, ax ):

    sub_X = generateSubBM(subset, X)
    #sub_X_2 = generateSubBM([6, 9, 11, 16, 18, 19, 21, 24, 29, 30, 31, 32, 35, 38, 39, 42, 43, 50, 51, 57, 58, 63, 65, 67, 68], X)
    config = classifier_obj['config']
    type_classifier = classifier_obj['type_classifier']
    name_classifier = classifier_obj['name_classifier']
    label = classifier_obj['label']
    COLOR = classifier_obj['color']
    tprs, mean_fpr = None, None
    if type_classifier == 'rf':
        tprs, mean_fpr = auc_roc_rf(sub_X, labels, config)
    elif type_classifier == 'knn':
        tprs, mean_fpr = auc_roc_knn(sub_X, labels, config)
    elif type_classifier == 'svm':
        tprs, mean_fpr = auc_roc_svm(sub_X, labels)
    elif type_classifier == 'nb':
        tprs, mean_fpr = auc_roc_nb(sub_X, labels)
    print('TYPE CLASSIFIER: ', type_classifier)
    print('tprs: ', len(tprs), ' mean_fpr: ', len(mean_fpr))
    mean_tpr = np.mean(tprs, axis=0, dtype=np.float64)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    #mean_tpr_2 = np.mean(tprs_2, axis=0, dtype=np.float64)
    #mean_tpr_2[-1] = 1.0
    #mean_auc_2 = auc(mean_fpr_2, mean_tpr_2)
    # Print the mean AUC value
    # Plot the mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color= COLOR, label=f'{label} = {mean_auc:.3f}')
    #fig.plot(mean_fpr_2, mean_tpr_2, color='orange', label=f'RF (t=100) - Mean AUC = {mean_auc_2:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')


def plot_pr_auc(subset, X, labels, classifier_obj, ax):
    sub_X = generateSubBM(subset, X)
    config = classifier_obj['config']
    type_classifier = classifier_obj['type_classifier']
    name_classifier = classifier_obj['name_classifier']
    label = classifier_obj['label']
    COLOR = classifier_obj['color']

    precisions, mean_recall = None, None
    if type_classifier == 'rf':
        precisions, mean_recall = pr_auc_rf(sub_X, labels, config)
    elif type_classifier == 'knn':
        precisions, mean_recall = pr_auc_knn(sub_X, labels, config)
    elif type_classifier == 'svm':
        precisions, mean_recall = pr_auc_svm(sub_X, labels)
    elif type_classifier == 'nb':
        precisions, mean_recall = pr_auc_nb(sub_X, labels)
    print('TYPE CLASSIFIER: ', type_classifier)
    print('precisions: ', len(precisions), ' mean_recall: ', len(mean_recall))
    mean_precision = np.mean(precisions, axis=0, dtype=np.float64)
    mean_precision[0] = 1.0
    mean_pr_auc = auc(mean_recall, mean_precision)

    ax.plot(mean_recall, mean_precision, color=COLOR, label=f'{label} = {mean_pr_auc:.3f}')


def readCsv(path):
    return  pd.read_csv(path, sep=',')

def generatePlots(plot = 'auc'):
    data, labels = processDataSet('train_normal_rtf.csv')
    path_classification = 'experiment_results/R3_v2/best_results_step_R3_v2.csv'
    classifiers_report = readCsv(path_classification)
    COLORS = sns.color_palette(n_colors=15)
    BASE_PATH = 'charts/third_step/'
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(classifiers_report)):
        subset = ast.literal_eval(classifiers_report.loc[i,'Subset'])
        name = classifiers_report.loc[i, 'Classifier']
        mh = classifiers_report.loc[i, 'Meta']
        print('MH: ', mh)
        of = classifiers_report.loc[i, 'OF']
        name_to_save = ''
        if plot == 'auc' or plot == 'both':
            config, name_structured, type_classifier = getClassifierName(name)
            dic_classifier = {
                'name_classifier': name_structured,
                'config': config,
                'type_classifier': type_classifier,
                'color': COLORS[i],
                'label': f'{function_map[of]} + {name_structured}'
            }
            ax.set_xlabel('False Positive Rate', fontname="Verdana",fontsize=25)
            ax.set_ylabel('True Positive Rate', fontname="Verdana",fontsize=25)
            plot_auc(subset, data, labels, dic_classifier, ax)
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
            ax.legend(loc="lower right", prop=font)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, 1)
            name_to_save = f'auc_journal_{mh}.png'
        elif plot == 'pr_vs_rec' or plot == 'both':
            config, name_structured, type_classifier = getClassifierName(name)
            dic_classifier = {
                'name_classifier': name_structured,
                'config': config,
                'type_classifier': type_classifier,
                'color': COLORS[i],
                'label': f'{function_map[of]} + {name_structured}'
            }
            # ({len(subset)}f)
            ax.set_xlabel('Recall', fontname="Verdana",fontsize=25)
            ax.set_ylabel('Precision', fontname="Verdana",fontsize=25)
            plot_pre_vs_rec(subset, data, labels, dic_classifier, ax)
            ax.legend(loc="lower left", prop=font)
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
            ax.set_ylim(0, 1.01)
            ax.set_xlim(0, 1)
            name_to_save = f'pre_vs_recall_journal_{mh}.png'
        elif plot == 'pr_auc':
            config, name_structured, type_classifier = getClassifierName(name)
            dic_classifier = {
                'name_classifier': name_structured,
                'config': config,
                'type_classifier': type_classifier,
                'color': COLORS[i],
                'label': f'{function_map[of]} + {name_structured}'
            }
            # ({len(subset)}f)
            ax.set_xlabel('Recall', fontname="Verdana",fontsize=25)
            ax.set_ylabel('Precision', fontname="Verdana",fontsize=25)
            plot_pr_auc(subset, data, labels, dic_classifier, ax)
            ax.legend(loc="lower left", prop=font)
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
            ax.set_ylim(0, 1.01)
            ax.set_xlim(0, 1)
            name_to_save = f'pr_auc_journal_{mh}.png'
        if (i + 1) % 5 == 0 :
            print(f'SAVED: {name_to_save}')
            plt.tight_layout()
            plt.savefig(f'{BASE_PATH}/{name_to_save}',dpi=700)
            fig, ax = plt.subplots(figsize=(8, 6))

if __name__ == "__main__":
    #main()
    #main_2()
    generatePlots('auc')
    generatePlots('pr_auc')