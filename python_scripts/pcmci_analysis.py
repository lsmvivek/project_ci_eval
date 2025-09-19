from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
import json
import sys
sys.path.append('../python_scripts')
from evaluate_adjacency_matrix import evaluate_adjacency_matrix, new_network_metric


def pcmci_analysis(dir_names, var_switch):
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
        for variable in variables_dict:
            variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.std(variables_dict[variable]))
        temp_dict={}
        temp_dict={k: v for k, v in variables_dict.items() if k not in excluded_variables}
        print(f'{len(temp_dict.keys())} for analysis')
        forcing_variables=[variable for variable in temp_dict.keys() if '_f_' in variable]
        forcing_variables.append('Rainf_tavg')
        forcing_variables_positions=[list(temp_dict.keys()).index(variable) for variable in forcing_variables]
        simulated_variables_positions=[index for index in list(range(len(temp_dict))) if index not in forcing_variables_positions]

        # Generate Gaussian white noise and add it to each variable in temp_dict
        np.random.seed(1111)
        for key in temp_dict:
            std_dev = np.std(temp_dict[key])
            noise = np.random.normal(0, 0.2 * std_dev, temp_dict[key].shape)
            temp_dict[key] += noise
        # PCMCI+ParCorr
        var_names = list(temp_dict.keys())
        data_numpy = np.stack([temp_dict[key] for key in temp_dict.keys()], axis=1);
        data_numpy=data_numpy[0:1*1000,:]
        dataframe = pp.DataFrame(data_numpy,
                                datatime = {0:np.arange(len(data_numpy[:,0]))}, 
                                var_names=var_names);
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(
            dataframe=dataframe, 
            cond_ind_test=parcorr,
            verbosity=1);
        tau_max=1
        tau_min=0;
        # Only estimate parents of variables 0, 1, 2
        link_assumptions = {}
        for j in range(len(temp_dict)):
            if j in simulated_variables_positions:
                # Directed lagged links
                link_assumptions[j] = {(var, -lag): '-?>' for var in simulated_variables_positions
                                for lag in range(1, tau_max + 1)}
                # Unoriented contemporaneous links
                link_assumptions[j].update({(var, 0): 'o?o' for var in simulated_variables_positions if var != j})
                # Directed lagged and contemporaneous links from the sun (3)
                link_assumptions[j].update({(var, -lag): '-?>' for var in forcing_variables_positions
                                for lag in range(0, tau_max + 1)})
            else:
                link_assumptions[j] = {}
        pc_alpha=0.05
        pcmci.verbosity = 0
        results_pcmciplus = pcmci.run_pcmciplus(tau_max=tau_max, tau_min= tau_min,       
                    link_assumptions=link_assumptions,
                    fdr_method='fdr_bh', pc_alpha=pc_alpha,);
        
        # Prepare adjacency matrix for evaluation
        # Define the mapping dictionary
        graph_mapping = {'': 0, '<--': -1, '-->': 1, 'o-o': 1, 'x-x': 1}
        adj_pcmci = results_pcmciplus['graph'].copy()
        # Apply the mapping to the graph
        adj_pcmci = np.vectorize(graph_mapping.get)(adj_pcmci)

        # Matrix with contemporaneous adjacencies
        B0=np.where(adj_pcmci[:,:,0] != 0, adj_pcmci[:,:,0] / np.abs(adj_pcmci[:,:,0]), 0);
        # Matrix with lagged adjacencies
        b1=np.where(adj_pcmci[:,:,1] != 0, adj_pcmci[:,:,1] / np.abs(adj_pcmci[:,:,1]), 0);
        ## Make the -1's as +1
        B0=np.where(B0 == -1, 1, B0)
        b1=np.where(b1 == -1, 1, b1)
        # Join the lagged and contemporaneous matrices
        B = np.concatenate([B0,b1], axis=0)

        metrics_dictionary=evaluate_adjacency_matrix(B, var_names, print_metrics=True, check_lag1=True);
        metrics_dictionary['my_network_metric']=new_network_metric(metrics_dictionary['tp'], metrics_dictionary['tn'], metrics_dictionary['fp'], metrics_dictionary['fn'])
        ## False direction of a true link
        pd.DataFrame(B, columns=var_names, index=var_names+[f'{var}_lag1' for var in var_names]).to_csv(f'../dircted_adj_results/pcmci_results/{dir}_abs_adj.csv', sep=',')
        with open(f'../dircted_adj_results/pcmci_results/{dir}.json', 'w') as f:
            json.dump(metrics_dictionary, f)
        print('finished', dir, 'saved in', f'../dircted_adj_results/pcmci_results/{dir}.json' '\n')

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
    pcmci_analysis(dir_names_1, var_switch=0)
    pcmci_analysis(dir_names_2, var_switch=1)