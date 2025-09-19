import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
import sys

def pcmciplus_function(data_dict: dict, test_parameters: dict):
    # print(data_dict)
    
    mci_test_name = test_parameters['mci_test_name']
    pc_alpha = test_parameters['pc_alpha']
    alpha_level = test_parameters['alpha_level']
    tau_max = int(test_parameters['tau_max'])
    tau_min = int(test_parameters['tau_min'])

    var_names = list(data_dict.keys())
    data_numpy = np.stack([data_dict[key] for key in data_dict.keys()], axis=1)
    # print(f'number of variables: {len(var_names)}', '\n', 'length of data:', {data_numpy.shape})
    dataframe = pp.DataFrame(data_numpy,
                         datatime = {0:np.arange(len(data_numpy[:,0]))}, 
                         var_names=var_names)
    if mci_test_name=='PC':
        cond_ind_test = ParCorr(significance='analytic')
    elif mci_test_name=='GPDC':
        cond_ind_test = GPDC(significance='analytic', gp_params=None)
    elif mci_test_name=='CMIknn':
        cond_ind_test = CMIknn(significance='shuffle_test', knn=0.2, \
                        shuffle_neighbors=5, transform='ranks', \
                    workers=40, sig_samples=200)
    pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=cond_ind_test,
    verbosity=0);

    # correlations = pcmci.run_bivci(tau_max=1, val_only=True)['val_matrix']
    # tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names, 'figsize':(15, 10),
    #                                     'x_base':1, 'y_base':.5}, )

    pcmci.verbosity = 0
    results_pcmciplus = pcmci.run_pcmciplus(tau_max=tau_max   , tau_min=tau_min,
            # contemp_collider_rule='Conservative', conflict_resolution=True, reset_lagged_links=False,     # 'Conservative', 'Majority'
            fdr_method='fdr_bh', pc_alpha=pc_alpha, )
    return results_pcmciplus