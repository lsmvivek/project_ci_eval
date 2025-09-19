import lingam
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
import json
import sys
sys.path.append('../python_scripts')
from evaluate_adjacency_matrix import evaluate_adjacency_matrix, new_network_metric, false_directions


def varlingam_analysis(dir_names, var_switch):
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
        else:
            excluded_variables=['SnowT_tavg', 'Swe_tavg', 'SnowDepth_tavg', 'Snowf_tavg', 'EvapSnow_tavg', 'Qsm_tavg', 'Qsb_tavg'];
        for variable in variables_dict:
            # variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.std(variables_dict[variable]))
            variables_dict[variable] = (variables_dict[variable] - np.mean(variables_dict[variable])) / (np.max(variables_dict[variable]) - np.min(variables_dict[variable]))

        temp_dict={}
        temp_dict={k: v for k, v in variables_dict.items() if k not in excluded_variables}
        print(f'{len(temp_dict.keys())} for analysis')
        # Generate Gaussian white noise and add it to each variable in temp_dict
        np.random.seed(1111)
        k=0.1
        for key in temp_dict:
            noise = np.random.gamma(shape=1, scale=k, size=temp_dict[key].shape)
            temp_dict[key] += noise
    
        var_names = list(temp_dict.keys())
        df = pd.DataFrame.from_dict(temp_dict, orient='columns')

        # VAR LiNGAM
        model = lingam.VARLiNGAM(lags=1, prune=True, criterion='bic', random_state=1111, lingam_model=None,)   #criterion{aic, fpe, hqic, bic, None}
        model.fit(df.values)

        # Matrix with contemporaneous adjacencies
        lingam_threshold_1=0.01
        lingam_threshold_2=0.1

        B0=np.where(abs(model.adjacency_matrices_[0]) > lingam_threshold_1, 1, 0).astype(bool);
        # Matrix with lagged adjacencies
        b1=np.where(abs(model.adjacency_matrices_[1]) > lingam_threshold_2, 1, 0);
        # Join the lagged and contemporaneous matrices
        B=np.concatenate([B0.T, b1.T], axis=0)

        metrics_dictionary=evaluate_adjacency_matrix(B, var_names, print_metrics=True, check_lag1=True);
        metrics_dictionary['my_network_metric']=new_network_metric(metrics_dictionary['tp'], metrics_dictionary['tn'], metrics_dictionary['fp'], metrics_dictionary['fn'])
        ## False direction of a true link
        # metrics_dictionary['false_dir'] = false_directions(a1, var_names)
        pd.DataFrame(B, columns=var_names, index=var_names+[f'{var}_lag1' for var in var_names]).to_csv(f'../dircted_adj_results/varlingam_results/{dir}_abs_adj.csv', sep=',')
        with open(f'../dircted_adj_results/varlingam_results/{dir}.json', 'w') as f:
            json.dump(metrics_dictionary, f)
        print('finished', dir, 'saved in', f'../dircted_adj_results/varlingam_results/{dir}.json' '\n')

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
    varlingam_analysis(dir_names_1, var_switch=0)
    varlingam_analysis(dir_names_2, var_switch=1)