
import os
from pathlib import Path
from pprint import pprint
from statistics import mean
from turtle import end_fill
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd
import pingouin as pg
import csv


def mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq, pivotData, remaniningData):

    columns = [ "Meta" , "OF","Length", "Settings", "Classifier",
                "F1_PIVOT", "F1_Compared", "Classifier_Compared",
                "Settings_Compared", "p-value", "RBC", "p-value-BH"]

    array = []
    p_values = []   # <-- store p-values

    for i, y in enumerate(ramining_models, start=1):
        y_mean = mean(y)
        U1, p_value = mannwhitneyu(pivot_model, y, alternative="two-sided", method="asymptotic")
        mwu_res = pg.mwu(pivot_model, y, alternative="two-sided")
        # Cálculo manual del Rank-Biserial Correlation (RBC)
        rbc = 1 - (2 * U1 / (len(pivot_model) * len(y)))
        values = [pivotData[0], pivotData[1], pivotData[2], pivotData[3],
                  pivotData[4], str(pivot_id), y_mean,
                  remaniningData[i - 1][0], remaniningData[i - 1][1],
                  p_value, rbc]

        array.append(values)
        p_values.append(p_value)  # <-- collect

    #Apply Benjamini-Hochberg correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    #Add corrected p-values to array
    for i in range(len(array)):
        array[i].append(pvals_corrected[i])

    return pd.DataFrame(data=array, columns=columns)

def vector_extraction_v2(pivot_id, df, meta, of):
    pivotValues = df[df['F1_MEAN'] == pivot_id]
    pivotData = [pivotValues.iloc[0]['Meta'], pivotValues.iloc[0]['OF'], pivotValues.iloc[0]['Length'], pivotValues.iloc[0]['Settings'], pivotValues.iloc[0]['Classifier']]
    pivotModel = pivotValues['F1'].to_numpy()

    remaningValues = df[df['F1'] != pivot_id]
    remaningModels = remaningValues['F1'].to_numpy()
    wrappedRemaningModels = [remaningModels[i:i + 30].tolist() for i in range(0, len(remaningModels), 30)]
    remainingData = []
    tempo = 0
    for _ in range(len(wrappedRemaningModels)):
        classifier = remaningValues.iloc[tempo]['Classifier']
        print(remaningValues.iloc[tempo]['Classifier'])
        settings = remaningValues.iloc[tempo]['Settings']
        remainingData.append([classifier, settings])
        tempo += 30
    print('PIVOT: ', len(pivotModel), ' - REMANING: ', len(wrappedRemaningModels) * len(wrappedRemaningModels[0]))
    return pivotModel, wrappedRemaningModels, pivotData, remainingData
if __name__ == "__main__":
    epochs=50
    check_freq=30
    print("------- Statistical test for the CNN model ----------")
    # model_history_path = 'histories/CNN/'
    # model_checkpoints_path = 'results/CNN_mean_checkpoints_history.csv'
    # Statistic_result_path = 'results/CNN_statistic_results.csv'
    # data = pd.read_csv(model_checkpoints_path)
    # pivot_id = data.iloc[data.idxmax()["val_auc"], :]["Epochs"]
    # pivot_model, ramining_models = vector_extraction(pivot_id, model_history_path, epochs)
    # mann_whitney(pivot_id, pivot_model, ramining_models, Statistic_result_path, check_freq)
    baseDir = 'results'
    pathMeanHistoryByFold = f'{baseDir}/meanFullHistoryByFold.csv'
    df = pd.read_csv(pathMeanHistoryByFold)
    metaArray = df['Meta'].unique()
    ofArray = df['OF'].unique()
    print('>> DF: ', df.columns)
    print('>> META ARRAY: ', metaArray)
    print('>> OF ARRAY: ', ofArray)
    static_dir = f'{baseDir}/statistic_results/'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    for meta in metaArray:
        for of in ofArray:
            print(f'>> Executing test for {meta} - {of}')
            filtered_df = df[(df['Meta'] == meta) & (df['OF'] == of)].reset_index()
            pivotValues = filtered_df.iloc[filtered_df["F1_MEAN"].idxmax(), :]
            pivotId = pivotValues['F1_MEAN']
            print('PIVOT ID: ', pivotId)
            resultsPath = f'{static_dir}/{meta}_{of}_{round(pivotId,2)}.csv'
            pivotModel, remaningModels, pivotData, remainingData = vector_extraction_v2(pivotId, filtered_df, meta, of)
            resultsDF = mann_whitney(pivotId, pivotModel, remaningModels, resultsPath, check_freq, pivotData, remainingData)
            resultsDF.to_csv(resultsPath, sep=',', index=False)