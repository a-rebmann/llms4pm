import pandas as pd
from bpmnjsonanalyzer import sanitize_label
from llms4pm.const import DATA_RESULTS


def preprocess_anomalies(anomalies):
    anomalies_per_type = {}
    for trace_id, anomalies in anomalies.items():
        anomalies_per_type[trace_id] = {0: [], 1: [], 2: []}
        for anomaly in anomalies:
            if anomaly[0] == 0:
                anomalies_per_type[trace_id][anomaly[0]].append(
                    [sanitize_label(label) for label in anomaly[1] if anomaly[1] != '']
                )
            else:
                if anomaly[1] != '':
                    anomalies_per_type[trace_id][anomaly[0]].append(sanitize_label(anomaly[1]))
    return anomalies_per_type


def evaluate(traces, detected_anomalies, true_anomalies):
    # for each trace, group anomalies per anomaly type and sanitize the labels
    detected_anomalies_per_type = preprocess_anomalies(eval(detected_anomalies))
    true_anomalies_per_type = preprocess_anomalies(eval(true_anomalies))
    # compute true positives, false positives and false negatives per anomaly type
    tp = {0: 0, 1: 0, 2: 0}
    fp = {0: 0, 1: 0, 2: 0}
    fn = {0: 0, 1: 0, 2: 0}
    for trace_id in detected_anomalies_per_type.keys():
        for anomaly_type in [0, 1, 2]:
            for detected_anomaly in detected_anomalies_per_type[trace_id][anomaly_type]:
                if detected_anomaly in true_anomalies_per_type[trace_id][anomaly_type]:
                    tp[anomaly_type] += 1
                else:
                    fp[anomaly_type] += 1
            for true_anomaly in true_anomalies_per_type[trace_id][anomaly_type]:
                if true_anomaly not in detected_anomalies_per_type[trace_id][anomaly_type]:
                    fn[anomaly_type] += 1
    # compute precision, recall and f1-score per anomaly type
    precision = {0: 0, 1: 0, 2: 0}
    recall = {0: 0, 1: 0, 2: 0}
    f1 = {0: 0, 1: 0, 2: 0}
    for anomaly_type in [0, 1, 2]:
        if tp[anomaly_type] + fp[anomaly_type] > 0:
            precision[anomaly_type] = tp[anomaly_type] / (tp[anomaly_type] + fp[anomaly_type])
        if tp[anomaly_type] + fn[anomaly_type] > 0:
            recall[anomaly_type] = tp[anomaly_type] / (tp[anomaly_type] + fn[anomaly_type])
        if precision[anomaly_type] + recall[anomaly_type] > 0:
            f1[anomaly_type] = 2 * (precision[anomaly_type] * recall[anomaly_type]) / (
                    precision[anomaly_type] + recall[anomaly_type])
    return precision[0], recall[0], f1[0], precision[1], recall[1], f1[1], precision[2], recall[2], f1[2]



GPT_MODEL = "gpt-3.5-turbo"
LLAMA_MODEL = "meta-llama-Llama-2-7b-chat-hf"
MISTRAL_MODEL = "mistralai-Mistral-7B-Instruct-v0.2"

MODEL_ID ="examples_cot_" + GPT_MODEL

DRIVE_RESULTS = "/Users/adrianrebmann/Google Drive/results/"


def evaluate_model():
    # load the results from drive
    results = pd.read_csv(DRIVE_RESULTS + MODEL_ID + ".csv")
    # load the original logs
    # eval_data = pd.read_csv(DATA_ROOT / "eval_data.csv")
    # join the results with the original logs on model_id and revision_id
    # eval_data = eval_data.merge(results, on=["model_id", "revision_id"])
    # evaluate per log
    results[["precision_out_of_order", "recall_out_of_order", "f_1_out_of_order", "precision_missing", "recall_missing",
             "f_1_missing", "precision_superfluous", "recall_superfluous", "f_1_superfluous"]] = results.apply(
        lambda x: pd.Series([*evaluate(eval(x["noisy"]), x["detected_anomalies"], x["labels"])]), axis=1)
    # save the results
    results.to_csv(DATA_RESULTS / (MODEL_ID + "_evaluated.csv"), index=False)


evaluate_model()
