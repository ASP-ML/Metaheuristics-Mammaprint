import ast
from utils import (processDataSet, os, datetime, pd)
import classifiers

R_K_TIMES = 1

def classifiers_evaluation(resume_file_path, data_file_name, parent_folder = None, child_folder = None, STEP_NAME= None):
    columns=['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'ACC_MEAN', 'ACC_DESV', 'PRE_MEAN', 'PRE_DESV', 'MSE_MEAN', 'MSE_DESV', 'REC_MEAN', 'REC_DESV', 'AUC_MEAN', 'AUC_DESV', 'F1_MEAN', 'F1_DESV', 'MCC_MEAN','MCC_DESV', 'SPEC_MEAN', 'SPEC_DESV']
    columns_matrix =['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'tn_mean', 'fp_mean',
                        'fn_mean', 'tp_mean', 'tn_desv', 'fp_desv', 'fn_desv', 'tp_desv', 'tn_sum', 'fp_sum',
                        'fn_sum', 'tp_sum']
    results_df = pd.DataFrame(columns=columns)
    matrix_df = pd.DataFrame(columns = columns_matrix)
    resume_table = pd.read_csv(resume_file_path, sep = ',')
    resume_table_np = resume_table.to_numpy()
    data, labels = processDataSet(f'{data_file_name}.csv')
    
    if parent_folder is not None and child_folder is not None:
        date_now = datetime.now()
        date_str = date_now.strftime('%Y_%m_%d')
        time_str = date_now.strftime('%H_%M_%S')
    else:
        date_str = parent_folder
        time_str = child_folder
    
    path = 'classifiers_results/' + data_file_name + '/' + date_str + '/' + time_str
    full_history_name_file = 'min_len_last_step_full_history.csv'
    if os.path.exists(path) == False:
        os.makedirs(path)

    with open(f'{path}/{full_history_name_file}', 'w') as file:
        file.write(f'Meta,OF,Generation,Length,Settings,Classifier,Fold,AUC,Accuracy,Precision,MSE,Recall,F1,MCC,Spec\n')
        
    for index, data_entry in enumerate(resume_table_np):
        print(f'>> Starting with {data_entry[1]}-{data_entry[3]}')
        best_subset = ast.literal_eval(data_entry[5])
        params = {
            'model': data_entry[1],
            'generation': data_entry[4],
            'settings': data_entry[2],
            'of_name': data_entry[3],
            'best_subset': data_entry[5],
            'length': len(best_subset),
            'root_path': path
        }
        print('PARAMS: ', params)
        weights_path = f'{path}/{data_entry[1]}/{index}_{data_entry[1]}_{data_entry[3]}_weights_{data_entry[3]}'
        if os.path.exists(weights_path) == False:
            os.makedirs(weights_path)
        # results_classifiers, matrix_results = classifiers.mlp_pipeline(best_subset, data, labels, R_K_TIMES, params, columns, columns_matrix ,f'{path}/{full_history_name_file}', weights_path)
        file = open(f'{path}/{full_history_name_file}', 'a')
        results_classifiers, matrix_results = classifiers.pipeline_updated(best_subset, data, labels, R_K_TIMES, params, columns, columns_matrix ,f'{path}/{full_history_name_file}', weights_path)
        file.close()
        results_df =  pd.concat([results_df, results_classifiers], ignore_index=True)
        matrix_df = pd.concat([matrix_df, matrix_results], ignore_index = True)
    results_df.to_csv(f'{path}/class_step_{STEP_NAME}_report_{data_file_name}.csv', sep=',', index = False)
    matrix_df.to_csv(f'{path}/matrix_step_{STEP_NAME}_report_{data_file_name}.csv', sep=',', index = False)

if __name__ == "__main__":
    base_path = 'results/default_date/default_time'
    last_report_path = f'{base_path}/total_history_1_report.csv'
    classifiers_evaluation(last_report_path, 'train_normal_rtf', parent_folder='default_date', child_folder='default_time', STEP_NAME='1')