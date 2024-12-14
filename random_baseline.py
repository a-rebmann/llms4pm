import random
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from const import DATA_ROOT
from eval_util import compute_footprint_fitness, compute_footprint_matrix_pairs
from evaluate_llm import get_discovery_data, get_prefix_data, get_trace_data, get_pair_data


def get_random_activity(x):
    s = eval(x) if isinstance(x, str) else x
    rand = random.randint(0, len(s) - 1)
    s = list(s)
    return s[rand]


def generate_random_matrix(activities):
    activities = sorted(activities)  # Sort to maintain order
    n = len(activities)
    # Step 2: Create an empty n x n matrix
    footprint_matrix = np.full((n, n), '#', dtype='<U2')  # Initialize with '#'
    for i in range(n):
        for j in range(n):
            ## pick an operator randomly from the list
            operators = ["#",'→']
            operator = random.choice(operators)
            footprint_matrix[i][j] = operator
    for i in range(n):
        for j in range(n):
            if footprint_matrix[i][j] == '→' and footprint_matrix[j][i] == '→':
                footprint_matrix[i][j] = footprint_matrix[j][i] = '‖'
            elif footprint_matrix[i][j] == '→' and footprint_matrix[j][i] == '#':
                footprint_matrix[j][i] = '←'

    return footprint_matrix


def run_random_df_and_pt(runs, sample_sizes):
    result_records = []
    for sample_size in sample_sizes:
        for run in range(runs):
            _, val_df, _ = get_discovery_data(sample_size)
            val_df["y_matrix"] = val_df["unique_activities"].apply(lambda x: generate_random_matrix(x))
            val_df["true_matrix"] = val_df.apply(lambda x: compute_footprint_matrix_pairs(x["dfg"], x["unique_activities"]), axis=1)
            
            fitness = list(val_df.apply(lambda x: compute_footprint_fitness(x["true_matrix"], x["y_matrix"]), axis=1).values)
            # Compute fintess
            rec = {
                "sample_size": sample_size,
                "run": run,
                "avg_fitness": mean(fitness),
            }
            result_records.append(rec)
    df = pd.DataFrame(result_records)
    df.to_csv(DATA_ROOT / "random_df_results.csv", index=False)


def run_random_nap(runs, sample_sizes):
    result_records = []
    for sample_size in sample_sizes:
        for run in range(runs):
            _, prefix_val_df, _ = get_prefix_data(sample_size)
            prefix_val_df["y"] = prefix_val_df["unique_activities"].apply(lambda x: get_random_activity(x))
            true_labels = prefix_val_df['next']
            predicted_labels = prefix_val_df['y']
            # Compute precision, recall, and F1 score
            precision_micro = precision_score(true_labels, predicted_labels, average='micro')
            recall_micro = recall_score(true_labels, predicted_labels, average='micro')
            f1_micro = f1_score(true_labels, predicted_labels, average='micro')
            precision_macro = precision_score(true_labels, predicted_labels, average='macro')
            recall_macro = recall_score(true_labels, predicted_labels, average='macro')
            f1_macro = f1_score(true_labels, predicted_labels, average='macro')
            rec = {
                "sample_size": sample_size,
                "run": run,
                "precision mic": precision_micro,
                "recall mic": recall_micro,
                "f1 mic": f1_micro,
                "precision mac": precision_macro,
                "recall mac": recall_macro,
                "f1 mac": f1_macro,
            }
            result_records.append(rec)
    df = pd.DataFrame(result_records)
    df.to_csv(DATA_ROOT / "random_nap_results.csv", index=False)


def run_random_trace_detection(runs, sample_sizes):
    result_records = []
    for sample_size in sample_sizes:
        for run in range(runs):
            _, val_df, _ = get_trace_data(sample_size)
            val_df["y"] = val_df.apply(lambda x: random.choices([True, False], k=1)[0], axis=1)
            true_labels = ~val_df['anomalous']
            predicted_labels = val_df['y']
            # Compute precision, recall, and F1 score
            precision_micro = precision_score(true_labels, predicted_labels, average='micro')
            recall_micro = recall_score(true_labels, predicted_labels, average='micro')
            f1_micro = f1_score(true_labels, predicted_labels, average='micro')
            precision_macro = precision_score(true_labels, predicted_labels, average='macro')
            recall_macro = recall_score(true_labels, predicted_labels, average='macro')
            f1_macro = f1_score(true_labels, predicted_labels, average='macro')
            rec = {
                "sample_size": sample_size,
                "run": run,
                "precision mic": precision_micro,
                "recall mic": recall_micro,
                "f1 mic": f1_micro,
                "precision mac": precision_macro,
                "recall mac": recall_macro,
                "f1 mac": f1_macro,
            }
            result_records.append(rec)
        df = pd.DataFrame(result_records)
        df.to_csv(DATA_ROOT / "random_t_sad_results.csv", index=False)


def run_random_out_of_order(runs, sample_sizes):
    result_records = []
    for sample_size in sample_sizes:
        for run in range(runs):
            _, val_df, _ = get_pair_data(sample_size)
            val_df["y"] = val_df.apply(lambda x: random.choices([True, False], k=1)[0], axis=1)
            true_labels = ~val_df['out_of_order']
            predicted_labels = val_df['y']
            # Compute precision, recall, and F1 score
            precision_micro = precision_score(true_labels, predicted_labels, average='micro')
            recall_micro = recall_score(true_labels, predicted_labels, average='micro')
            f1_micro = f1_score(true_labels, predicted_labels, average='micro')
            precision_macro = precision_score(true_labels, predicted_labels, average='macro')
            recall_macro = recall_score(true_labels, predicted_labels, average='macro')
            f1_macro = f1_score(true_labels, predicted_labels, average='macro')
            rec = {
                "sample_size": sample_size,
                "run": run,
                "precision mic": precision_micro,
                "recall mic": recall_micro,
                "f1 mic": f1_micro,
                "precision mac": precision_macro,
                "recall mac": recall_macro,
                "f1 mac": f1_macro,
            }
            result_records.append(rec)
        df = pd.DataFrame(result_records)
        df.to_csv(DATA_ROOT / "random_a_sad_results.csv", index=False)


if __name__ == '__main__':
    #run_random_trace_detection(runs=3, sample_sizes=[10000])
    #run_random_out_of_order(runs=3, sample_sizes=[10000])
    #run_random_nap(runs=3, sample_sizes=[10000])
    run_random_df_and_pt(runs=3, sample_sizes=[500])
