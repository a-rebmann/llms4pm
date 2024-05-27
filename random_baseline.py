import random

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from const import DATA_ROOT
from evaluate_llm import get_prefix_data, get_trace_data, get_pair_data


def get_random_activity(x):
    s = eval(x) if isinstance(x, str) else x
    rand = random.randint(0, len(s) - 1)
    s = list(s)
    return s[rand]


def run_random_nap(runs, sample_sizes):
    result_records = []
    for sample_size in sample_sizes:
        for run in range(runs):
            prefix_train_df, prefix_val_df, prefix_test_df = get_prefix_data(sample_size)
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
            train_df, val_df, test_df = get_trace_data(sample_size)
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
            train_df, val_df, test_df = get_pair_data(sample_size)
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
    run_random_trace_detection(runs=5, sample_sizes=[10000])
    run_random_out_of_order(runs=5, sample_sizes=[10000])
    run_random_nap(runs=5, sample_sizes=[10000])

