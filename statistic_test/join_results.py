import pandas as pd

if __name__ == "__main__":
    baseDir = 'results'
    combined_statistic_results_path = f'{baseDir}/combined_statistic_results.csv'
    mean_by_fold_path = f'{baseDir}/meanByFold.csv'
    df = pd.read_csv(combined_statistic_results_path)
    df_mean_by_fold = pd.read_csv(mean_by_fold_path)
    
    df_combined = pd.merge(df, df_mean_by_fold, on=['Meta', 'OF', 'Length', 'Settings', 'Classifier'], how='inner')
    df_combined.to_csv(f'{baseDir}/combined_statistic_results_with_mean_by_fold.csv', index=False)
    print('>> Combined DataFrame with mean by fold saved to: ', f'{baseDir}/combined_statistic_results_with_mean_by_fold.csv')