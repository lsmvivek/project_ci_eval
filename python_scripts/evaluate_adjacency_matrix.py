import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, \
    f1_score, accuracy_score, confusion_matrix, matthews_corrcoef, roc_curve, auc, fbeta_score
from scipy.spatial.distance import hamming
import scipy.linalg as slin


def zheng2018(_w_mat, d_vars):
    return np.trace(slin.expm(_w_mat * _w_mat)) - d_vars


def evaluate_adjacency_matrix(pred_matrix, pred_vars_list, true_matrix=None, print_metrics=True, check_lag1=True):
    # Load and filter the true adjacency matrix
    if true_matrix is None:
        true_matrix = pd.read_csv('/home/caliber/research/ci_eval/data/adjacency_matrix/benchmark_GLDAS_tru_directed_adj_no_snow_with_lag1.csv', index_col=0)

        if check_lag1: true_matrix = true_matrix.loc[pred_vars_list+[f'{var}_lag1' for var in pred_vars_list], pred_vars_list]  # cause in rows and effects in columns
        else: true_matrix = true_matrix.loc[pred_vars_list, pred_vars_list]

    true_matrix = true_matrix.values
    if isinstance(pred_matrix, pd.DataFrame) and check_lag1 is False:
        pred_matrix = pred_matrix.loc[pred_vars_list, pred_vars_list].values
    elif isinstance(pred_matrix, np.ndarray) and check_lag1 is False:
        pred_matrix = pred_matrix[0:len(pred_vars_list), :]

    true_flat = true_matrix.flatten()
    pred_flat = pred_matrix.flatten()

    # Calculate performance metrics
    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0,)
    fbeta = fbeta_score(true_flat, pred_flat, beta=2, zero_division=0)

    # Confusion matrix: [TN, FP, FN, TP]
    tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Balanced Accuracy
    balanced_acc = (recall + specificity) / 2

    # MCC
    mcc=matthews_corrcoef(true_flat, pred_flat)

    tpr = tp/(tp + fn) if (tp + fn) > 0 else 0
    fpr = fp/(fp + tn) if (fp + tn) > 0 else 0
    hamming_distance = hamming(true_flat, pred_flat)

    # frobenius_norm=np.linalg.norm(true_matrix.astype(int)-pred_matrix.astype(int), ord='fro')
    # Print the metrics if requested
    if print_metrics:
        print(f"Performance Metrics: tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"F2 Score: {fbeta:.2f}")
        print(f"True Positive Ratio (TPR): {tpr:.2f}")
        print(f"False Positive Ratio (FPR): {fpr:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"Balanced Accuracy: {balanced_acc:.2f}")
        print(f"MCC: {mcc:.2f}")
        # print(f"Frobenius Norm: {frobenius_norm:.2f}")
        print(f"Hamming Distance: {hamming_distance:.2f}")
    
    metrics_dict={'accuracy': accuracy, 'precision': precision, 'recall': recall,
                  'f1': f1, 'fbeta': fbeta, 'tpr': tpr, 'fpr': fpr, 'specificity': specificity,
                  'balanced_acc': balanced_acc, 'mcc': mcc,
                  'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                  'hamming_distance': hamming_distance,
                #   'frobenius_norm': frobenius_norm}
    }
    for key in metrics_dict:
        metrics_dict[key]=float(round(metrics_dict[key], 3))

    return metrics_dict

# list(df.columns)
def new_network_metric(tp,tn,fp,fn):
    print(f'(tp+tn)/(fp+fn): {((tp+tn)/(fp+fn)):.2f}')
    print(f'tp + tn: {tp+tn}')
    return round((tp+tn)/(fp+fn), 3)