import numpy as np
import pandas as pd
import HydroErr as he
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append('./an_ml_exercise/python_scripts')
from get_bigdf import get_bigdf
import os
os.chdir('./an_ml_exercise')
##############################################################################################################
target_ = 'SoilMoist_S_tavg'
dir_ = '82.875_27.625'  

# Prediction time period
train_start_date = '2000-01-01' 
train_end_date = '2003-12-31'
test_start_date = '2004-01-01'
test_end_date = '2005-12-31'

# PCC
target_predictors_pcc = ['ACond_tavg', 'CanopInt_tavg', 'ECanop_tavg', 'ESoil_tavg', 'Evap_tavg', 
                            'GWS_tavg', 'Lwdown_f_tavg', 'Lwnet_tavg', 'Qair_f_tavg', 'Qg_tavg',
                            'Qh_tavg', 'Qle_tavg', 'Qs_tavg', 'Qsb_tavg', 'Rainf_f_tavg', 
                            'Rainf_tavg', 'SoilMoist_P_tavg', 'SoilMoist_RZ_tavg', 'Swdown_f_tavg',
                            'Swnet_tavg', 'Tveg_tavg', 'Tws_tavg', 'Wind_f_tavg',
                            'ACond_tavg_lag1', 'CanopInt_tavg_lag1', 'ECanop_tavg_lag1',
                            'ESoil_tavg_lag1', 'Evap_tavg_lag1', 'GWS_tavg_lag1', 'Lwdown_f_tavg_lag1', 
                            'Lwnet_tavg_lag1', 'Qair_f_tavg_lag1', 'Qg_tavg_lag1', 'Qh_tavg_lag1',
                            'Qle_tavg_lag1', 'Qs_tavg_lag1','Qsb_tavg_lag1', 'Rainf_f_tavg_lag1',
                            'Rainf_tavg_lag1', 'SoilMoist_P_tavg_lag1', 'SoilMoist_RZ_tavg_lag1',
                            'SoilMoist_S_tavg_lag1', 'Swdown_f_tavg_lag1', 'Swnet_tavg_lag1',
                            'Tveg_tavg_lag1', 'Tws_tavg_lag1', 'Wind_f_tavg_lag1']
##############################################################################################################
dates = [train_start_date, train_end_date, test_start_date, test_end_date]
big_df = get_bigdf(target_, dir_, dates)

# Monte Carlo simulation parameters
MONTE_CARLO_ITERATIONS = 100
NOISE_LEVELS = [0, 0.1, 0.2, 0.5, 1.0]  # Different noise levels to test
RANDOM_SEED = 42
##############################################################################################################
# Model architecture and parameters
# # SVR with RBF kernel
svr_rbf = make_pipeline(SVR(kernel='rbf'))      # rbf poly linear
param_grid = {'svr__C': [0.1, 1, 10],'svr__epsilon': [1e-6, 1e-4, 1e-2],
              'svr__gamma': ['scale', 'auto']}
grid_search = GridSearchCV(svr_rbf, param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)
##############################################################################################################

def monte_carlo_sims(target_predictors, method):
    np.random.seed(RANDOM_SEED)
    monte_carlo_results_training = {'method': [], 'noise_level': [], 'r2_score': [], 'rmse': [], 'mae': [], 'nse': [], 'nse_mod': [], 'kge': []}
    monte_carlo_results = {'method': [], 'noise_level': [], 'r2_score': [], 'rmse': [], 'mae': [], 'nse': [], 'nse_mod': [], 'kge': []}
    pred_ts = []
    for noise_level in NOISE_LEVELS:
        print(f"Noise level: {noise_level}")
        for _ in range(MONTE_CARLO_ITERATIONS):
            method = method            #      PCC      VARLiNGAM      PCMCIplus
            monte_carlo_results['method'].append(method)
            monte_carlo_results_training['method'].append(method)
            k = noise_level
            monte_carlo_results['noise_level'].append(k)
            monte_carlo_results_training['noise_level'].append(k)
            X = big_df[target_predictors].copy()
            y = big_df[target_].copy() #* grid_area / density_of_water  # m/d

            X_train, y_train = X.loc[train_start_date:train_end_date].values, y.loc[train_start_date:train_end_date].values
            X_test, y_test = X.loc[test_start_date:test_end_date].values, y.loc[test_start_date:test_end_date].values


            # Add gaussian white noise for realistic scenario
            noise_train = np.random.normal(0, k*np.std(X_train, axis=0), size=X_train.shape, )
            X_train_noisy = X_train + noise_train
            y_train_noisy = y_train + np.random.normal(0, k*np.std(y_train), size=y_train.shape)
            # Add gaussian white noise for realistic scenario
            noise_test = np.random.normal(0, k*np.std(X_test, axis=0), size=X_test.shape)
            X_test_noisy = X_test + noise_test
            y_test_noisy = y_test + np.random.normal(0, k*np.std(y_test), size=y_test.shape)

            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train_noisy)
            X_test_scaled = scaler_X.transform(X_test_noisy)
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train_noisy.reshape(-1, 1)).flatten()

            # Fit model with true features only
            model_train = grid_search.fit(X_train_scaled, y_train_scaled)

            train_y_predicted = grid_search.predict(X_train_scaled)
            train_y_predicted = scaler_y.inverse_transform(train_y_predicted.reshape(-1, 1)).flatten()

            test_y_predicted  = grid_search.predict(X_test_scaled)
            test_y_predicted  = scaler_y.inverse_transform(test_y_predicted.reshape(-1, 1)).flatten()

            # Save all the results in a dictionary
            train_results = {
                "r2_score": he.r_squared(observed_array=y_train_noisy, simulated_array=train_y_predicted),
                "rmse": he.rmse(observed_array=y_train_noisy, simulated_array=train_y_predicted),
                "mae": he.mae(observed_array=y_train_noisy, simulated_array=train_y_predicted),
                "nse": he.nse(observed_array=y_train_noisy, simulated_array=train_y_predicted),
                "nse_mod": he.nse_mod(observed_array=y_train_noisy, simulated_array=train_y_predicted),
                "kge": he.kge_2009(observed_array=y_train_noisy, simulated_array=train_y_predicted)
            }

            # Save all the results in a dictionary
            test_results = {
                "r2_score": he.r_squared(observed_array=y_test_noisy, simulated_array=test_y_predicted),
                "rmse": he.rmse(observed_array=y_test_noisy, simulated_array=test_y_predicted),
                "mae": he.mae(observed_array=y_test_noisy, simulated_array=test_y_predicted),
                "nse": he.nse(observed_array=y_test_noisy, simulated_array=test_y_predicted),
                "nse_mod": he.nse_mod(observed_array=y_test_noisy, simulated_array=test_y_predicted),
                "kge": he.kge_2009(observed_array=y_test_noisy, simulated_array=test_y_predicted)
            }
            monte_carlo_results_training['r2_score'].append(train_results['r2_score'])
            monte_carlo_results_training['rmse'].append(train_results['rmse'])
            monte_carlo_results_training['mae'].append(train_results['mae'])
            monte_carlo_results_training['nse'].append(train_results['nse'])
            monte_carlo_results_training['nse_mod'].append(train_results['nse_mod'])
            monte_carlo_results_training['kge'].append(train_results['kge'])

            monte_carlo_results['r2_score'].append(test_results['r2_score'])
            monte_carlo_results['rmse'].append(test_results['rmse'])
            monte_carlo_results['mae'].append(test_results['mae'])
            monte_carlo_results['nse'].append(test_results['nse'])
            monte_carlo_results['nse_mod'].append(test_results['nse_mod'])
            monte_carlo_results['kge'].append(test_results['kge'])
            pred_ts.append(test_y_predicted)
    
    # Save the results to a file
    monte_carlo_df = pd.DataFrame(monte_carlo_results)
    monte_carlo_df_training = pd.DataFrame(monte_carlo_results_training)
    pred_ts_array = np.array(pred_ts)
    # save to dir
    np.save(f'data/save_results/{method}-MC-pred_ts_{target_}_{dir_}.npy', pred_ts_array)
    monte_carlo_df.to_csv(f'data/save_results/{method}-MC-results_{target_}_{dir_}.txt', index=False)
    monte_carlo_df_training.to_csv(f'data/save_results/{method}-MC-results_{target_}_{dir_}_training.txt', index=False)
    return None

##############################################################################################################

predictors_list = [target_predictors_pcc]
method_list = ['PCC']

# predictors_list = [target_predictors_pcc, target_predictors_tcdf, target_predictors_varlingam, target_predictors_pcmciplus, target_predictors_dynotears]
# method_list = ['PCC', 'TCDF', 'VARLiNGAM', 'PCMCIplus', 'DYNOTEARS']
if __name__ == "__main__":
    for method_, target_predictors_ in zip(method_list, predictors_list):
        contemp_predictors = [pred_ for pred_ in target_predictors_ if '_lag1' not in pred_]
        lagged_predictors = [pred_ for pred_ in target_predictors_ if '_lag1' in pred_]
        print(f'\n{"="*60}')
        print(f'Running MC simulations for: {method_}')
        print(f'Contemporary predictors: {len(contemp_predictors)}')
        print(f'Lagged predictors: {len(lagged_predictors)}')
        print(f'Total predictors: {len(target_predictors_)}')
        print(f'{"="*60}')
        
        monte_carlo_sims(target_predictors=target_predictors_, method=method_)