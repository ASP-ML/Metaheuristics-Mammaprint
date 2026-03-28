import pandas as pd
import os

def concat_statistic_results(base_dir):
    statistic_result_path = f'{base_dir}/statistic_results/'
    list_dir = os.listdir(statistic_result_path)
    df = pd.DataFrame()
    for file_name in list_dir:
        df = pd.concat([df, pd.read_csv(f'{statistic_result_path}/{file_name}')])
        print(f'>> DF for {file_name}: ', df)

    # Benjamini-Hochberg correction
    df['Comparison BH'] = df['p-value-BH'].apply(lambda x: 'p < 0.05' if x < 0.05 else x)
    df['Comparison'] = df['p-value'].apply(lambda x: 'p < 0.05' if x < 0.05 else x)

    df.to_csv(f'{base_dir}/combined_statistic_results.csv', index=False)
    print('>> Combined DataFrame: ', df)
    
if __name__ == "__main__":
    base_dir = 'results'
    concat_statistic_results(base_dir)
    print('>> Finished concatenating statistic results.')
    