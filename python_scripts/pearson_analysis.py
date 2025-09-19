from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
import json
import sys
sys.path.append('/home/caliber/research/dynamical_model/python_scripts/')
sys.path.append('../python_scripts')
from evaluate_adjacency_matrix import evaluate_adjacency_matrix, new_network_metric, false_directions


# Function to compute Pearson correlation matrix
def compute_pearson_correlation_matrix(df):
    cols = df.columns
    corr_matrix = pd.DataFrame(np.zeros([len(cols), len(cols)]), index=cols, columns=cols)
    corr_matrix_lag1 = pd.DataFrame(np.zeros([len(cols), len(cols)]), index=cols, columns=cols)
    threshold_=0.2
    significane_threshold=0.05
    for col1 in cols:
        for col2 in cols:
            # Check if lagged correlation is present and significant
            corr, p_val = pearsonr(df[col1].iloc[:-1], df[col2].iloc[1:])
            if corr > threshold_ and p_val < significane_threshold: corr_matrix_lag1.loc[col1, col2] = 1
            elif corr < -1*threshold_ and p_val < significane_threshold: corr_matrix_lag1.loc[col1, col2] = 1 # -1
            # Check if normal correlation is significant
            # Make diagonals 0, since autocorrelation is now handled by corr_matrix_lag1
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 0
            else:
                corr, p_val = pearsonr(df[col1], df[col2])
                if corr > threshold_ and p_val < significane_threshold: corr_matrix.loc[col1, col2] = 1
                elif corr < -1*threshold_ and p_val < significane_threshold: corr_matrix.loc[col1, col2] = 1 # -1
                # If not significant, set to 0
                else: corr_matrix.loc[col1, col2] = 0
                # print(col1, col2, corr_matrix.loc[col1, col2])
    return pd.concat([corr_matrix, corr_matrix_lag1], axis=0).values.astype(int)

def pearson_adj_analysis(dir_names, var_switch):
    for dir in dir_names:
        file_list=sorted(glob.glob(f'/storage/extracted_grids/{dir}/*.txt'));    
        print('starting', dir)
        variables_dict={}
        for file in file_list:
            temp=pd.read_csv(file, sep=',', index_col=0)
            variables_dict[temp.columns[0]]=temp.iloc[:,0].values
            del temp
        variables_dict = OrderedDict(sorted(variables_dict.items()));
        variables_to_change_units_celcius=['Tair_f_tavg', 'AvgSurfT_tavg']
        for variable in variables_to_change_units_celcius:
            variables_dict[variable]=variables_dict[variable]-273.15
        if var_switch == 0:
            excluded_variables=['SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg', 'EvapSnow_tavg', 'Qsm_tavg',];
        elif var_switch == 1:
            excluded_variables=['SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg', 'EvapSnow_tavg', 'Qsm_tavg', 'Qsb_tavg'];
        elif var_switch == 2:
            excluded_variables=['Qsb_tavg', 'ECanop_tavg', 'ESoil_tavg', 'EvapSnow_tavg', 'Evap_tavg', 'Tveg_tavg']
        
        # Normalize the variables in variables_dict
        for variable in variables_dict:
            variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.max(variables_dict[variable]) - np.min(variables_dict[variable]))
        
        temp_dict={}
        temp_dict={k: v for k, v in variables_dict.items() if k not in excluded_variables}
        print(f'{len(temp_dict.keys())} for analysis')

        # Create the dataframe
        var_names = list(temp_dict.keys());
        df = pd.DataFrame.from_dict(temp_dict, orient='columns').iloc[0:]
        var_names = list(df.keys())
        # Compute the correlation matrix
        correlation_matrix = compute_pearson_correlation_matrix(df);
        
        # Evaluate the adjacency matrix
        ## Now calculate the standard metrics
        correlation_matrix_abs=np.where(correlation_matrix==-1, 1, correlation_matrix)
        metrics_dictionary=evaluate_adjacency_matrix(correlation_matrix_abs, var_names, print_metrics=True, check_lag1=True);
        metrics_dictionary['my_network_metric']=new_network_metric(metrics_dictionary['tp'], metrics_dictionary['tn'], metrics_dictionary['fp'], metrics_dictionary['fn'])
        pd.DataFrame(correlation_matrix, columns=var_names, index=var_names+[f'{var}_lag1' for var in var_names]).to_csv(f'../dircted_adj_results/pearson_results/{dir}_abs_adj.csv', sep=',')
        with open(f'../dircted_adj_results/pearson_results/{dir}.json', 'w') as f:
            json.dump(metrics_dictionary, f)
        print('finished', dir, 'saved in', f'../dircted_adj_results/pearson_results/{dir}.json' '\n')

if __name__ == '__main__':
    dir_names_1=['78.875_24.375', '82.125_23.875', '82.875_27.625', '79.125_23.875','88.875_25.375', 
            '-64.625_0.875', '-73.125_-6.875', '-60.125_-2.375', '-60.625_-4.125', '-66.625_-3.125',
            '-53.625_-3.625', '-64.125_-6.875', '-55.125_1.375', '-59.875_-7.125', '-72.125_-10.125',
            '26.125_46.875', '18.875_48.625', '15.325_47.625', '19.875_43.325',
            '16.625_45.875',  '17.625_46.625',  '20.325_44.325', '26.325_42.325',
            '-92.875_37.125', '-93.375_32.625', '-86.625_35.125', '-82.875_37.125',
            '-92.875_39.875', '-95.625_42.625', '-88.625_40.125', '-92.875_43.825', '-98.325_42.125']
    # Turn off Qsb_avg for below with var_switch=1
    dir_names_2 = ['140.125_-33.125',  '143.125_-32.375', '145.125_-29.625', '144.375_-33.625', '144.375_-28.125',
                '148.125_-29.125', '146.625_-31.375', '146.875_-25.875', '147.125_-27.375', '150.875_-28.125',
                '16.875_49.375',  '-98.125_34.625' ,  '23.125_43.325' ]
    dir_names_snow=['-49.625_66.125', '-44.625_62.125', '-52.625_77.625', '-45.125_75.625', '-35.625_69.625']
    pearson_adj_analysis(dir_names_1, var_switch=0)
    pearson_adj_analysis(dir_names_2, var_switch=1)