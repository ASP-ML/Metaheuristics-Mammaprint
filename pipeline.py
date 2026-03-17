# Standard library imports
import multiprocessing
import time
import gc
import concurrent.futures
# Third-party imports
import pandas as pd
# Local imports
import ofunctions
from utils import (make_experiment_dirs, processDataSet, Experiment, getMetasByObjectiveFunction, ExecutorWrapper)

def run_experiment_using_weka(mh_dict,
                              data,
                              labels,
                              STEPS_EXPERIMENT,
                              FIRST_GEN,
                              END_GEN,
                              STEP_GEN,
                              NUMBER_EXECUTORS,
                              METAS_EXPERIMENT,
                              OBJECTIVE_FUNCTIONS,
                              base_path,
                              historic_path):
    for i in range(STEPS_EXPERIMENT):
        print(f">>>> Starting step experiment: {i}/{STEPS_EXPERIMENT}")
        #Here mh_dict is generated with all data
        keys_dict = list(mh_dict.keys())
        n_run = 1
        report_generated_list_path = []
        for gen in range(FIRST_GEN, END_GEN + 1, STEP_GEN):
            print(f">>> Starting epoch experiment: {gen}/{END_GEN}")
            ctx = multiprocessing.get_context('spawn')
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
                    print(f'> Step {i + 1}-({gen}/{END_GEN}): Futures availables: {executors_free}/{NUMBER_EXECUTORS}')
                    print(f'> Step {i + 1}-({gen}/{END_GEN}): Process assigned: {process_assigned}/{total_process}')
                    if process_assigned == (total_process):
                        completed_all = True
                    if completed_all == False and executors_free > 0:
                        remaining_mh = []
                        for mh in mh_dict:
                            if mh_dict[mh].processAssigned() == False:
                                remaining_mh.append(mh_dict[mh])
                        print(f'> Step {i + 1}-({gen}/{END_GEN}): Remaning mhs evaluation: {total_process - len(remaining_mh)}/{total_process}')
                        for free, mh in zip(list(range(executors_free)), remaining_mh):
                            executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                            meta = mh.metaheuristic
                            mh.setProcessAssigned(True)
                            future = executor.submit(meta.runProcess)
                            time.sleep(10)
                            futures.append(future)
                            futures_wrapper.append(ExecutorWrapper(executor, future))
                            process_assigned += 1
                        print(f'> Step {i + 1}-({gen}/{END_GEN}): Updated mhs evaluation: {total_process - len(remaining_mh)}/{total_process}')
                    time.sleep(5)
                print(f'>> Got {len(results_futures)} results')
                for params in results_futures:
                    experiment.addDataByMeta(params, i)
                n_run += 1
                for key in mh_dict:
                    mh_dict[key].metaheuristic.set_new_iterations(n_run * STEP_GEN)
                    mh_dict[key].setDone(False)
                    mh_dict[key].setProcessAssigned(False)
                print(f">>> Finish Epoch experiment: {gen}/{END_GEN}")
            total_historic = experiment.getHistoricByStep(i)
            total_historic.to_csv(f'{base_path}/total_history_{i + 1}_report.csv')
            historic_by_step, historic_by_step_df = experiment.getBestByObjectiveFunction(i, get_df=True)
            report_generated_list_path.append(f'{base_path}/step_{i + 1}_report.csv')
            historic_by_step_df.to_csv(len(report_generated_list_path) - 1, index = False)
            mh_dict = getMetasByObjectiveFunction(data,labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, FIRST_GEN, historic_by_step, path_report=historic_path)
    return report_generated_list_path

    
if __name__ == "__main__":
    print("--------------- Metaheuristic Collection  ---------------")
    results = []
    data_set = None
    subset = []
    data = None
    labels = None
    objective_functions_list = [
                                ofunctions.mi_sklearn, 
                                ofunctions.mi_sklearn_2, 
                                ofunctions.mi_prop]
    
    STEPS_EXPERIMENT = 1
    START_GEN = 0
    END_GEN = 100
    STEP_GEN = 10
    NUMBER_EXECUTORS = 4
    METAS_EXPERIMENT = ['GA', 'SA', 'DE']
    OBJECTIVE_FUNCTIONS = ['relieff', 'mi', 'sim_uncertainty', 'gain_ratio', 'information_gain', 'chi2']
    columns=['Meta', 'OF', 'IT','Subset', 'Length', 'Settings','Classifier', 'ACC_MEAN', 'ACC_DESV', 'PRE_MEAN', 'PRE_DESV', 'MSE_MEAN', 'MSE_DESV', 'REC_MEAN', 'REC_DESV', 'AUC_MEAN', 'AUC_DESV', 'F1_MEAN', 'F1_DESV', 'MCC_MEAN','MCC_DESV']
    R_K_TIMES = 1
    _date = "25_03_2024"
    _time = "12_00_00"
    
    data, labels = processDataSet('train_normal_rtf.csv')
    base_path, historic_path = make_experiment_dirs(_date, _time, auto_time_stamp=False)

    experiment = Experiment(STEPS_EXPERIMENT, len(METAS_EXPERIMENT))
    results_df = pd.DataFrame(columns=columns)
    first_gen = START_GEN + STEP_GEN
    mh_dict = getMetasByObjectiveFunction(data,labels, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, first_gen, path_report=historic_path)
    report_generated_list_path = run_experiment_using_weka(mh_dict, data, labels, STEPS_EXPERIMENT, first_gen, END_GEN, STEP_GEN, NUMBER_EXECUTORS, METAS_EXPERIMENT, OBJECTIVE_FUNCTIONS, base_path, historic_path)
    print('>> Experiment finished. Report generated in: ', report_generated_list_path)