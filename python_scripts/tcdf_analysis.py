import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
import json
import sys

from scipy import optimize
sys.path.append('./python_scripts/')
from evaluate_adjacency_matrix import evaluate_adjacency_matrix, new_network_metric, false_directions
sys.path.append('./gitrepos/TCDF/')
# from runTCDF_mod import runTCDF
import subprocess

def tcdf_analysis(dir_names, var_switch):
    for dir in dir_names:
        file_list=sorted(glob.glob(f'./extracted_grids/{dir}/*.txt'));    
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
        else:
            excluded_variables=['SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg', 'EvapSnow_tavg', 'Qsm_tavg', 'Qsb_tavg'];
        for variable in variables_dict:
            variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.max(variables_dict[variable]) - np.min(variables_dict[variable]))
            variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.std(variables_dict[variable]))
        temp_dict={}
        temp_dict={k: v for k, v in variables_dict.items() if k not in excluded_variables}
        print(f'{len(temp_dict.keys())} for analysis')

        # Create the dataframe
        var_names = list(temp_dict.keys())
        df = pd.DataFrame.from_dict(temp_dict, orient='columns')
        df = df.iloc[0:4000,:]        

        # 2025-01-28, new implementation
        # df = df / df.max()
        
        # Define the arguments explicitly
        data_arg="--data"
        file_path='./dircted_adj_results/temp_tcdf/{dir}.csv'
        df.to_csv(file_path, sep=',', index=False)

        kernel_arg="--kernel_size"
        kernel_size="4"

        dialation_arg="--dilation_coefficient"
        dilation_coefficient="4"

        hidden_layers_arg="--hidden_layers"
        hidden_layers="0"

        significance_arg="--significance"
        significance = "1"

        learning_rate_arg="--learning_rate"
        learning_rate="0.005"

        cuda_arg="--cuda"

        optimizer_arg="--optimizer"
        optimizer="RMSprop"
        
        # Pass the arguments to the script
        subprocess.run(["/home/caliber/miniconda3/envs/bhagwan/bin/python", "/home/caliber/research/gitrepos/TCDF/runTCDF.py", 
                                data_arg, file_path, kernel_arg, kernel_size, dialation_arg, dilation_coefficient, hidden_layers_arg, hidden_layers,
                                significance_arg, significance,  cuda_arg, optimizer_arg, optimizer], text=True, capture_output=True)

        # 24-02-2025, new implementation
        #  Create the adjacency matrix using the alldealys.txt file
        with open('./dircted_adj_results/temp_tcdf/alldelays.txt', 'r') as f:
            lines = f.readlines()   # # (cause, effect): delay
        alldelays = {}
        # split to remove commas and brackets, into cause, effect and delay
        for line in lines:
            key, value = line.strip().split(': ')
            alldelays[key] = value
        # create a list of tuples of cause, effect and delay
        cause_effect_delay_tuple = []
        for key, value in alldelays.items():
            cause_effect_delay_tuple.append((key.split(': ')[0].split(', ')[-1].split(')')[0], key.split(': ')[0].split(', ')[0].split('(')[1], value))
        # print(cause_effect_delay_tuple[0], '\n')
        # now create a numpy adjacency matrix
        contemporaneous_adj= np.zeros([df.shape[1],df.shape[1]])
        lagged_adj= np.zeros([df.shape[1],df.shape[1]])
        for cause_, effect_, lag_ in cause_effect_delay_tuple:
            if lag_=='0':
                contemporaneous_adj[int(cause_), int(effect_)]=1
            elif lag_=='1':
                lagged_adj[int(cause_), int(effect_)]=1
            else: pass
                # print(f'more lags {lag_}') 
        # Create a combined adjacency matrix
        combined_adj=np.concatenate((contemporaneous_adj, lagged_adj), axis=0)
        # Evaluate the adjacency matrix
        metrics_dictionary=evaluate_adjacency_matrix(combined_adj, var_names, print_metrics=True, check_lag1=True);
        metrics_dictionary['my_network_metric']=new_network_metric(metrics_dictionary['tp'], metrics_dictionary['tn'], metrics_dictionary['fp'], metrics_dictionary['fn'])
        
        pd.DataFrame(combined_adj, columns=var_names, index=var_names+[f'{var}_lag1' for var in var_names]).to_csv(f'../dircted_adj_results/tcdf_results/{dir}_abs_adj.csv', sep=',')
        with open(f'./dircted_adj_results/tcdf_results/{dir}.json', 'w') as f:
            json.dump(metrics_dictionary, f)
        print('finished', dir, 'saved in', f'./dircted_adj_results/tcdf_results/{dir}.json' '\n')


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
    tcdf_analysis(dir_names_1, var_switch=0)
    tcdf_analysis(dir_names_2, var_switch=1)