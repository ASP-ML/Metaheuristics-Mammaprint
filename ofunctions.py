import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn import metrics
import math
from skrebate import ReliefF
import numpy as np
from weka.core.dataset import create_instances_from_matrices
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection
from weka.filters import Filter

stratifiedKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)

def generateSubBM(features, BM):
    if len(features) == 0:
        features = [0]
    n, _ = BM.shape
    subBM = np.zeros((n, len(features)), dtype= np.float64)
    for idx, value in enumerate(features):
        subBM[:, idx] = BM[:, value].reshape(n,).astype(np.float64)
    return subBM

def fitnessFunctionGWO(positions, data, labels, k = 7, kFolds = 10):
    features = np.where(positions>=0.4999)[0]
    train_data = generateSubBM(features, data)
    auc_list = []
    knn = KNeighborsClassifier(k)
    stratifiedKFold = StratifiedKFold(kFolds, shuffle=True, random_state=45)
    for train_index, test_index in stratifiedKFold.split(train_data, labels):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        knn.fit(X_train, y_train)
        y_probability = knn.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_probability[:,1])
        auc_list.append(auc_score)
    w = 0.9
    return (w*np.mean(auc_list) + (1-w) * 1/(len(positions)))

def fitnessFunctionDE(positions, data, labels, k = 7, kFolds = 10):
    features = np.where(positions>=0.4999)[0]
    train_data = generateSubBM(features, data)
    auc_list = []
    knn = KNeighborsClassifier(k)
    stratifiedKFold = StratifiedKFold(kFolds, shuffle=True, random_state=45)
    for train_index, test_index in stratifiedKFold.split(train_data, labels):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        knn.fit(X_train, y_train)
        y_probability = knn.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_probability[:,1])
        auc_list.append(auc_score)
    return np.mean(auc_list)

def fitnessFunctionSA(positions, data, labels, k = 7, kFolds = 10):
    train_data = generateSubBM(positions, data)
    auc_list = []
    knn = KNeighborsClassifier(k)
    stratifiedKFold = StratifiedKFold(kFolds, shuffle=True, random_state=45)
    for train_index, test_index in stratifiedKFold.split(train_data, labels):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        knn.fit(X_train, y_train)
        y_probability = knn.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_probability[:,1])
        auc_list.append(auc_score)
    return np.mean(auc_list)

def entropy(y):
    probs = [] # Probabilities of each class label
    unique = set(y)
    for c in unique: # Set gets a unique set of values. We're iterating over each value
        num_same_class = 0
        for i in y:
            if c == i:
                num_same_class += 1 # Remember that true == 1, so we can sum.
        p = num_same_class / len(y) # Probability of this class label
        probs.append(p)
    return np.sum([-p * np.log2(p) for p in probs])


def class_probability(feature, y, length):
    """Calculates the proportional length of each value in the set of instances"""
    probs = []
    for value in set(feature):
        select = feature == value # Split by feature value into two classes
        y_new = y[select]         # Those that exist in this class are now in y_new
        probs.append(float(len(y_new))/length)  # Convert to float, because ints don't divide well
    return probs
def class_entropy(feature, y):
    """Calculates the entropy for each value in the set of instances"""
    ents = []
    for value in set(feature):
        select = feature == value # Split by feature value into two classes
        y_new = y[select]         # Those that exist in this class are now in y_new
        ents.append(entropy(y_new))
    return ents

def informationGain(subset, labels, entropyDataSet):
    """Entropy(S) - Sum |Sv|/|S| Entropy(Sv)"""
    nCols = subset.shape[0]
    probs = class_probability(subset, labels, nCols) #For compute the probability of features
    ents = class_entropy(subset, labels) #For compute the entropy of features
    entropyFeature =  np.sum(np.multiply(probs, ents))
    return entropyDataSet - entropyFeature

def entropyByColumn(positions, data):
    features = np.where(positions >= 0.4999)[0]
    subMatrix = generateSubBM(features, data)
    entropyList = []
    for i in subMatrix.T:
        entropyList.append(entropy(i))
    return np.sum(entropyList) / len(entropyList)
        
def entropyByColumn2(positions, data):
    subMatrix = generateSubBM(positions, data)
    entropyList = []
    for i in subMatrix.T:
        entropyList.append(entropy(i))
    return np.sum(entropyList) / len(entropyList)

def mutual_information(data, x_index, labels, debug = False):
    """
    Calculate and return Mutual information between two random variables
    """
    n_rows, n_cols = data.shape
    # Check if index are into the bounds
    assert (x_index >= 0 and x_index <= n_rows)
    # Variable to return MI
    summation = 0.0
    # Get uniques values of random variables
    values_x = set(data[x_index])
    values_y = set(labels)
    # Print debug info
    if debug:
        print('MI between')
        # print(data[x_index])
        # print(labels)
        print('Index: ', x_index)
    # For each random
    for value_x in values_x:
        for value_y in values_y:
            px = np.shape(np.where(data[x_index]==value_x))[1] / n_cols
            py = np.shape(np.where(labels==value_y))[1] / n_cols
            pxy = len(np.where(np.in1d(np.where(data[x_index]==value_x)[0], 
                            np.where(labels==value_y)[0])==True)[0]) / n_cols
            if pxy > 0.0:
                summation += pxy * math.log((pxy / (px*py)))
            # if debug:
            #     print ('(%f,%f) px:%f py:%f pxy:%f' % (value_x, value_y, px, py, pxy))
    return summation

def mi(positions, data, labels):
    mi_array = []
    data = data.T
    for i in positions: 
        mi_array.append(mutual_information(data, i, labels))
    return np.sum(mi_array)

def mi_sklearn(positions, data, labels):
    subMatrix = generateSubBM(positions, data)
    array = []
    for i in range(len(subMatrix[0])):
        score = mutual_info_classif(subMatrix[:, [i]], labels, discrete_features=[False], random_state=45)
        array.append(score[0])
    return sum(array)
def mi_sklearn_by_submatrix(subMatrix, labels):
    array = []
    for i in range(len(subMatrix[0])):
        score = mutual_info_classif(subMatrix[:, [i]], labels, random_state=45)
        array.append(score[0])
    return array
def mi_sklearn_2(positions, data, labels):
    """Calculate the difference between the total information of dataset and the mutual information of a subset\n
    PARAMETERS 
    positions: indexes of dataset features
    data: dataset as numpy array
    labels: dataset labels"""
    total_mi = mutual_info_classif(data, labels, random_state=123)
    subMatrix = generateSubBM(positions, data)
    score = mutual_info_classif(subMatrix, labels, random_state= 123)
    return sum(total_mi) - sum(score)

def mi_prop(positions, data, labels):
    total_mi = mutual_info_classif(data, labels, random_state=123)
    subMatrix = generateSubBM(positions, data)
    score = mutual_info_classif(subMatrix, labels, random_state= 123)
    # print('Total mi: ', sum(total_mi), ' ', ' subset mi: ', sum(score), ' div: ', sum(score) / sum(total_mi))
    return sum(score) / sum(total_mi)

def svmFitnessValue(positions, data, labels):
    clf = SVC(kernel= 'poly', probability=True)
    subMatrix = generateSubBM(positions, data)
    aucList = []
    for train_index, test_index in stratifiedKFold.split(subMatrix, labels):
        X_train, X_test = subMatrix[train_index], subMatrix[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        y_probability = clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(y_test,  y_probability[:, 1])
        aucList.append(auc)
    return np.mean(aucList)

def knnFitnessValue(positions, data, labels, k = 7, kFolds = 10):
    train_data = generateSubBM(positions, data)
    auc_list = []
    knn = KNeighborsClassifier(k)
    stratifiedKFold = StratifiedKFold(kFolds, shuffle=True, random_state=45)
    for train_index, test_index in stratifiedKFold.split(train_data, labels):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        knn.fit(X_train, y_train)
        y_probability = knn.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_probability[:,1])
        auc_list.append(auc_score)
    return np.mean(auc_list)

def gain_ratio(subMatrix, labels, information_gain):
    array = []
    for i in range(subMatrix.shape[1]):
        feature = subMatrix[:, [i]]
        clf_ig = DecisionTreeClassifier(criterion='entropy')
        clf_ig.fit(feature, labels)
        samples_by_node = clf_ig.tree_.n_node_samples
        intrinsic_info = - np.sum((samples_by_node[1:] / samples_by_node[0]) * np.log2(samples_by_node[1:] / samples_by_node[0]))
        array.append(intrinsic_info)
    array_gain = np.divide(information_gain, array)
    return array_gain

def entropy_by_column(train_data):
    array = []
    for i in train_data:
        array.append(entropy(train_data[:,i]))
    return sum(array)

def simmetical_uncertainty_by_subset(data, labels):
    entropy_labels = entropy(labels)
    array = []
    for i in range(data.shape[1]):
        feature = data[:, [i]]
        mutual_info = mutual_info_classif(feature, labels, random_state=45)[0]
        entropy_data = entropy(feature.squeeze())
        array.append((2 * mutual_info) / (entropy_data + entropy_labels))
    return array

def informationGainBySubset(subsetMatrix, labels):
    entropy_labels = entropy(labels)
    array = []
    for i in range(subsetMatrix.shape[1]):
        ig = informationGain(subsetMatrix[:,i], labels, entropy_labels)
        array.append(ig)
    return array

def allObjectiveFunctions(positions, data, labels, type_of = '', get_mean = True):
    train_data = generateSubBM(positions, data)

    if type_of == 'sim_uncertainty' or type_of == 'information_gain' or type_of == 'gain_ratio' or type_of == 'mi':
        if type_of == 'mi':
            mutual_info = mi_sklearn_by_submatrix(train_data, labels)
            if get_mean:
                return sum(mutual_info) / len(mutual_info)
            return mutual_info
        elif type_of == 'sim_uncertainty':
            su = simmetical_uncertainty_by_subset(train_data, labels)
            if get_mean:
                return sum(su) / len(su)
            return su
        elif type_of == 'information_gain' or type_of == 'gain_ratio':
            ig = informationGainBySubset(train_data, labels)
            if type_of == 'information_gain':
                if get_mean:
                    return ig
                return sum(ig) / len(ig)
            elif type_of == 'gain_ratio':
                gr = gain_ratio(train_data, labels, ig)
                if get_mean:
                    return gr
                return sum(gr) / len(gr)
    elif type_of == 'relieff':
        #RELIEFF
        r = ReliefF()
        r.fit(train_data, labels)
        relieff = r.feature_importances_
        if get_mean:
            return sum(relieff) / len(relieff)
        return relieff
        
def allObjectiveFunctionWeka(positions, data, labels, type_of = '', get_mean = True):
    try:
        if len(positions) == 0:
            return 0
        train_data = generateSubBM(positions, data)
        if train_data.shape[1] == 0:
            return 0
        if type_of == 'mi':
            mutual_info = mi_sklearn_by_submatrix(train_data, labels)
            del train_data
            if get_mean:
                return sum(mutual_info) / len(mutual_info)
            return mutual_info
        elif type_of == 'sim_uncertainty':
            su = simmetrical_uncertainty_weka(train_data, labels, False)
            mean = np.sum(su[:,1]) / train_data.shape[1]
            del train_data
            if get_mean:
                return mean
            return su
        elif type_of == 'gain_ratio':
            gr = gain_ratio_weka(train_data, labels, False)
            mean = np.sum(gr[:,1]) / train_data.shape[1]
            del train_data
            if get_mean:
                return mean
            return gr
        elif type_of == 'information_gain':
            ig = information_gain_weka(train_data, labels, False)
            mean = np.sum(ig[:,1]) / train_data.shape[1]
            del train_data
            if get_mean:
                return mean
            return ig
        elif type_of == 'relieff':
            rl = relieff_weka(train_data, labels, False)
            mean = np.sum(rl[:,1]) / train_data.shape[1]
            del train_data
            if get_mean:
                return mean
            return rl
        elif type_of == 'chi2':
            chi = chi2_weka(train_data, labels, False)
            
            mean = np.sum(chi[:,1]) / train_data.shape[1]
            del train_data
            if get_mean:
                return mean
            return chi
    except Exception as e:
        print(e)
        print('LENGTH: ', len(positions))
        print('Positions: ', positions)
        return 0
    
def information_gain_weka(data, labels, get_mean = True):
    dataset = create_instances_from_matrices(data, labels, name="generated from matrices")
    # print(dataset)
    # Apply NumericToNominal filter
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    filter.inputformat(dataset)
    filtered_instances = filter.filter(dataset)
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(filtered_instances)
    dict_results = attsel.ranked_attributes
    
    if get_mean:
        mean = np.mean(dict_results[:, 1])
        return mean
    return dict_results
    


def gain_ratio_weka(data, labels, get_mean = True):
    dataset = create_instances_from_matrices(data, labels, name="generated from matrices")
    # print(dataset)
    # Apply NumericToNominal filter
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    filter.inputformat(dataset)
    filtered_instances = filter.filter(dataset)
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.GainRatioAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(filtered_instances)
    dict_results = attsel.ranked_attributes
    
    if get_mean:
        mean = np.mean(dict_results[:, 1])
        return mean
    return dict_results

def simmetrical_uncertainty_weka(data, labels, get_mean = True):
    dataset = create_instances_from_matrices(data, labels, name="generated from matrices")
    # print(dataset)
    # Apply NumericToNominal filter
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    filter.inputformat(dataset)
    filtered_instances = filter.filter(dataset)
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.SymmetricalUncertAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(filtered_instances)
    dict_results = attsel.ranked_attributes
    
    if get_mean:
        mean = np.mean(dict_results[:, 1])
        return mean
    return dict_results

def relieff_weka(data, labels, get_mean = True):
    dataset = create_instances_from_matrices(data, labels, name="generated from matrices")
    # print(dataset)
    # Apply NumericToNominal filter
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    filter.inputformat(dataset)
    filtered_instances = filter.filter(dataset)
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.ReliefFAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(filtered_instances)
    dict_results = attsel.ranked_attributes
    
    if get_mean:
        mean = np.mean(dict_results[:, 1])
        return mean
    return dict_results

def chi2_weka(data, labels, get_mean = True):
    
    dataset = create_instances_from_matrices(data, labels, name="generated from matrices")
    # print(dataset)
    # Apply NumericToNominal filter
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    filter.inputformat(dataset)
    filtered_instances = filter.filter(dataset)
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-N", "-1"])
    evaluation = ASEvaluation(classname="weka.attributeSelection.ChiSquaredAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluation)
    attsel.select_attributes(filtered_instances)
    dict_results = attsel.ranked_attributes
    
    if get_mean:
        mean = np.mean(dict_results[:, 1])
        return mean
    return dict_results