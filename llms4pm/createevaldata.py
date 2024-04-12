import random
from copy import deepcopy

import pandas as pd

from llms4pm.behavioral_profile import get_behavioral_profile_as_df
from llms4pm.verifier import is_true_anomaly
from langdetect import DetectorFactory, detect, LangDetectException
from const import DATA_ROOT
# Set a seed for reproducibility (optional)
DetectorFactory.seed = 0

NOISY_TRACE_PROB = 0.8
NOISY_EVENT_PROB = 0.5
LOG_SIZE = 10


def _insert_event(trace: list, tasks: set):
    if len(trace) <= 1:
        return trace, []
    ins_index = random.randint(0, len(trace))
    task = random.choice(list(tasks))
    trace.insert(ins_index, task)
    return trace, [task]


def _swap_events(trace: list):
    if len(trace) <= 1:
        return trace, []
    swap_index = random.randint(0, len(trace) - 2)
    trace2 = trace.copy()
    trace2[swap_index], trace2[swap_index + 1] = trace2[swap_index + 1], trace2[swap_index]
    return trace2, [trace2[swap_index], trace2[swap_index + 1]]


def _remove_event(trace: list):
    if len(trace) <= 1:
        return trace, []
    del_index = random.randint(0, len(trace) - 1)
    trace2 = list()
    deleted_event = None
    for i in range(0, len(trace)):
        if i != del_index:
            trace2.append(trace[i])
        else:
            deleted_event = trace[i]
    return trace2, [deleted_event]


def _get_event_classes(log):
    classes = set()
    for trace in log:
        for task in trace:
            classes.add(task)
    return classes


def check_and_add_anomaly(trace, affected, trace_idx_to_noise, idx, noise_type, strict_order, reverse_strict_order,
                          exclusive, interleaving):
    if len(affected) >= 1 and is_true_anomaly(noise_type, affected, trace, strict_order,
                                             reverse_strict_order, exclusive, interleaving):
        # check if the other anomalies still hold for this trace after the noise was inserted
        for anomaly in trace_idx_to_noise[idx]:
            if not is_true_anomaly(anomaly[0], anomaly[1], trace, strict_order,
                                   reverse_strict_order,
                                   exclusive, interleaving):
                trace_idx_to_noise[idx].remove(anomaly)
        trace_idx_to_noise[idx].append((noise_type, affected))


def insert_noise(log, noisy_trace_prob, noisy_event_prob=0.0, log_size=10):
    strict_order, reverse_strict_order, exclusive, interleaving = get_behavioral_profile_as_df(log)
    if len(log) < log_size:
        # add additional traces until desired log size reached
        log_cpy = deepcopy(log)
        for i in range(0, log_size):
            log_cpy.append(deepcopy(log[i % len(log)]))
            if len(log_cpy) >= log_size:
                break
        log = log_cpy
    else:
        log = deepcopy(log)
    classes = _get_event_classes(log)
    log_new = list()
    trace_idx_to_noise = dict()
    for idx, trace in enumerate(log):
        trace_idx_to_noise[idx] = []
        if len(trace) > 0:
            trace_cpy = trace.copy()
            # check if trace makes random selection
            if random.random() <= noisy_trace_prob:
                insert_more_noise = True
                while insert_more_noise:
                    # randomly select which kind of noise to insert
                    noise_type = random.randint(0, 2)
                    if noise_type == 0:
                        trace_cpy, affected = _swap_events(trace_cpy)
                        check_and_add_anomaly(trace_cpy, affected, trace_idx_to_noise, idx, noise_type, strict_order,
                                              reverse_strict_order, exclusive, interleaving)
                    if noise_type == 1:
                        trace_cpy, affected = _insert_event(trace_cpy, classes)
                        check_and_add_anomaly(trace_cpy, affected, trace_idx_to_noise, idx, noise_type, strict_order,
                                              reverse_strict_order, exclusive, interleaving)
                    if noise_type == 2:
                        trace_cpy, affected = _remove_event(trace_cpy)
                        check_and_add_anomaly(trace_cpy, affected, trace_idx_to_noise, idx, noise_type, strict_order,
                                              reverse_strict_order, exclusive, interleaving)
                    # flip coin to see if more noise will be inserted
                    insert_more_noise = (random.random() <= noisy_event_prob)
            if len(trace_cpy) > 1:
                log_new.append(trace_cpy)
    return log_new, trace_idx_to_noise


def create_eval_data(df):
    df[["noisy", "labels"]] = df.apply(
        lambda x: pd.Series([*insert_noise(x["string_traces"], NOISY_TRACE_PROB, NOISY_EVENT_PROB, LOG_SIZE)]), axis=1)
    return df


def detect_language(x):
    try:
        res = detect(str(x).replace("{", "").replace("}", ""))
    except LangDetectException:
        print(f"Language detection failed for {x}")
        res = "unknown"
    return res



if __name__ == "__main__":
    model_df = pd.read_csv(DATA_ROOT / "basic_cleaned_corpus.csv")
    model_df["string_traces"] = model_df["string_traces"].apply(eval)
    model_df["unique_activities"] = model_df["string_traces"].apply(lambda x: set([e for t in x for e in t]))
    #detect model language
    model_df["language"] = model_df["unique_activities"].apply(lambda x: detect_language(x))
    create_eval_data(model_df)
    model_df.to_csv(DATA_ROOT / "eval_data.csv", index=False)
