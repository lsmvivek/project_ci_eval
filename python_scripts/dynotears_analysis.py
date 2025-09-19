from causalnex.structure import dynotears
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
import json
import sys
sys.path.append('../python_scripts')
from evaluate_adjacency_matrix import evaluate_adjacency_matrix
from evaluate_adjacency_matrix import new_network_metric

def dynotears_analysis(dir_names, var_switch):
    for dir in dir_names:
        file_list=sorted(glob.glob(f'/storage/extracted_grids/{dir}/*.txt'));    
        print('starting', dir)
        variables_dict={}
        for file in file_list:
            temp=pd.read_csv(file, sep=',', index_col=0)
            variables_dict[temp.columns[0]]=temp.iloc[:,0].values
            del temp
        variables_dict = OrderedDict(sorted(variables_dict.items()));
        if var_switch == 0:
            excluded_variables=['SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg', 'EvapSnow_tavg', 'Qsm_tavg',];
        elif var_switch == 1:
            excluded_variables=['SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg', 'EvapSnow_tavg', 'Qsm_tavg', 'Qsb_tavg'];
        elif var_switch == 2:
            excluded_variables=['Qsb_tavg', 'ECanop_tavg', 'ESoil_tavg', 'EvapSnow_tavg', 'Evap_tavg', 'Tveg_tavg']
        
        # Normalize the variables in variables_dict
        for variable in variables_dict:

            variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.std(variables_dict[variable]))
        temp_dict={}
        temp_dict={k: v for k, v in variables_dict.items() if k not in excluded_variables}
        forcing_variables=[variable for variable in temp_dict.keys() if '_f_' in variable]
        print(f'{len(temp_dict.keys())} for analysis')

        # Create the dataframe
        df = pd.DataFrame.from_dict(temp_dict, orient='columns')
        var_names = list(temp_dict.keys())
        df = df.iloc[0:,:]
        # dynotears
        # Create and run the model
        sm=dynotears.from_pandas_dynamic(df,
                                   tabu_child_nodes=forcing_variables,
                                   p=1,
                                max_iter=1*1000,
                                lambda_a=0.01,
                                lambda_w=0.001,
                                w_threshold=0.01,
                                h_tol=0.01)
        
        # Create the empty adjacency matrix
        W_contemporaneous=pd.DataFrame(np.zeros([len(df.columns), len(df.columns)], dtype=int), columns=df.columns, index=df.columns);
        A_lagged=pd.DataFrame(np.zeros([len(df.columns), len(df.columns)], dtype=int), columns=df.columns, index=[f'{var}_lag1' for var in df.columns]);
        # Define the mapping dictionary and apply
        for key1, value1 in sm.adj.items():
            # Matrix with contemporaneous adjacencies
            if key1.split('_')[-1]=='lag0':
                parent_ = '_'.join(key1.split('_')[:-1])
                if value1:
                    for key2 in value1:
                        child_ = '_'.join(key2.split('_')[:-1])
                        if value1[key2]['weight']:
                            W_contemporaneous.loc[parent_, child_] = 1  # value1[key2]['weight']    # 1
            # Matrix with lagged adjacencies
            elif key1.split('_')[-1]=='lag1':
                parent_=key1
                if value1:
                    for key2 in value1:
                        child_ = '_'.join(key2.split('_')[:-1])
                        if value1[key2]['weight']:
                            A_lagged.loc[parent_, child_] = 1  # value1[key2]['weight']    # 1
        W_contemporaneous=W_contemporaneous.values
        A_lagged=A_lagged.values
        ## Make the -1's as +1
        W_contemporaneous=np.where(W_contemporaneous == -1, 1, W_contemporaneous)
        A_lagged=np.where(A_lagged == -1, 1, A_lagged)
        # Join the lagged and contemporaneous matrices
        W_A_combined = np.concatenate([W_contemporaneous,A_lagged], axis=0)
        # Evaluate the adjacency matrix
        metrics_dictionary=evaluate_adjacency_matrix(W_A_combined, var_names, print_metrics=True, check_lag1=True);
        metrics_dictionary['my_network_metric']=new_network_metric(metrics_dictionary['tp'], metrics_dictionary['tn'], metrics_dictionary['fp'], metrics_dictionary['fn']);

        
        pd.DataFrame(W_A_combined, columns=var_names, index=var_names+[f'{var}_lag1' for var in var_names]).to_csv(f'../dircted_adj_results/dynotears_results/{dir}_abs_adj.csv', sep=',')
        with open(f'../dircted_adj_results/dynotears_results/{dir}.json', 'w') as f:
            json.dump(metrics_dictionary, f)
        print('finished', dir, 'saved in', f'../dircted_adj_results/dynotears_results/{dir}.json' '\n')

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
    dynotears_analysis(dir_names_1, var_switch=0)
    dynotears_analysis(dir_names_2, var_switch=1)