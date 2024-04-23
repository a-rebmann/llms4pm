import warnings
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from const import MISTRAL_MODEL, EVAL_PATH, TASKS
from llm.prompts import general_task_prompt_order, get_few_shot_prompt_pairs, general_task_prompt, \
    get_few_shot_prompt_traces


warnings.filterwarnings("ignore")
tqdm.pandas()


def split_by_model(df, test_size=0.5, random_state=4):
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    model_ids = df['id'].unique()
    train_ids, test_ids = train_test_split(model_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['id'].isin(train_ids)]
    test_df = df[df['id'].isin(test_ids)]
    return train_df, test_df


def get_model_and_tokenizer(model_name, device="cuda:1", quant_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )  # , quantization_config=bnb_config,
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_output(model_name, device, model, tokenizer, prompt):
    if model_name == MISTRAL_MODEL:
        prompt = "[INST]" + prompt + "[/INST]"
    #print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
    )
    first_token_probs = outputs.scores[0][0]
    true = tokenizer("True")["input_ids"]
    false = tokenizer("False")["input_ids"]
    # option_scores = (
    #     first_token_probs[[true, false]].float().cpu().numpy()
    # )  # True, False
    # print(option_scores)
    # pred = np.array(["True", "False"])[np.argsort(option_scores)[::-1][:1]]
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    print(tokenizer.decode(generated_tokens[0][0]), true, false)
    return tokenizer.decode(generated_tokens[0][0])


def run_evaluation_loop(model_name, device, model, tokenizer, prompt_sample_sizes, n_runs, train_df, val_df,
                        task_prompt, get_zero_shot_prompt=None, get_few_shot_prompt=None, input_att=None):
    if get_zero_shot_prompt is None and get_few_shot_prompt is None:
        print("Either a zero or few shot prompt generation function must be passed")
        return None
    if not input_att:
        input_att = "trace"
    result_records = []
    for sample_size in prompt_sample_sizes:
        done = False
        for run in range(n_runs):
            if sample_size == 0 and done:
                continue
            if get_zero_shot_prompt is None:
                this_prompt = get_few_shot_prompt(train_df, sample_size, task_prompt, input_att)
            else:
                this_prompt = get_zero_shot_prompt(task_prompt, input_att)
            val_df["y"] = val_df.progress_apply(lambda x: generate_output(model_name, device, model, tokenizer,
                                                                          this_prompt +
                                                                          str(x["unique_activities"]) + "\n" +
                                                                          "First activity, second activity: " +
                                                                          str(x[input_att]) + "\n"), axis=1)
            print("Detected raw", val_df['y'].value_counts())
            val_df['y'] = val_df['y'].apply(lambda x: True if x == 'True' else False)
            print("Detected parsed", val_df['y'].value_counts())
            if input_att == "trace":
                true_labels = val_df['anomalous']
            else:
                true_labels = ~val_df['anomalous']
            print("True parsed", true_labels.value_counts())
            predicted_labels = val_df['y']
            tp = sum((true_labels == True) & (predicted_labels == True))
            fp = sum((true_labels == False) & (predicted_labels == True))
            fn = sum((true_labels == True) & (predicted_labels == False))
            # Compute precision, recall, and F1 score
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            print(f"Sample size: {sample_size}, Run: {run}, TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision}, "
                  f"Recall: {recall}, F1: {f1}")
            result_records.append({
                "sample_size": sample_size,
                "run": run,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            done = True
    df = pd.DataFrame(result_records)
    return df


bnb_config = (BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
))


def get_pair_data(samples_per_class, with_trace=False):
    pair_df = pd.read_pickle(EVAL_PATH / "eval_train_data_pairs.pkl")
    if not with_trace:
        pair_df = pair_df.drop_duplicates(subset=["revision_id", "model_id", "eventually_follows"])
    pair_df["anomalous"] = pair_df["out_of_order"]
    # Set test_size to 0.5 to get 50% of the rows in the val set
    pair_train_df, pair_val_df = split_by_model(pair_df, test_size=0.5, random_state=4)
    # Sample data to have equal number of positive and negative samples
    pair_val_df_positive = pair_val_df[~pair_val_df["anomalous"]].sample(n=samples_per_class, random_state=4)
    pair_val_df_negative = pair_val_df[pair_val_df["anomalous"]].sample(n=samples_per_class, random_state=4)
    pair_val_df = pd.concat([pair_val_df_negative, pair_val_df_positive])
    # shuffle the data
    pair_val_df = pair_val_df.sample(frac=1, random_state=4).reset_index(drop=True)
    return pair_train_df, pair_val_df


def get_trace_data(samples_per_class):
    trace_df = pd.read_pickle(EVAL_PATH / "eval_train_data_traces.pkl")
    trace_df["anomalous"] = trace_df.progress_apply(lambda x: len(x["label"]) > 0, axis=1)
    # Set test_size to 0.5 to get 50% of the rows in the val set
    trace_train_df, trace_val_df = split_by_model(trace_df, test_size=0.5, random_state=4)
    # Sample data to have equal number of positive and negative samples
    trace_val_df_positive = trace_val_df[~trace_val_df["anomalous"]].sample(n=samples_per_class)
    trace_val_df_negative = trace_val_df[trace_val_df["anomalous"]].sample(n=samples_per_class)
    trace_val_df = pd.concat([trace_val_df_negative, trace_val_df_positive])
    # shuffle the data
    trace_val_df = trace_val_df.sample(frac=1).reset_index(drop=True)
    return trace_train_df, trace_val_df


def main():
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/work/arebmann/.chache/"
    parser = argparse.ArgumentParser(description='A simple command line interface')
    parser.add_argument('task', choices=TASKS, help='The task to run')
    parser.add_argument('device', help='Device to use (e.g. cuda:1)')
    parser.add_argument('model', help='The huggingface model to use (e.g. "google/gemma-7b-it")')
    parser.add_argument('prompt_sample_sizes', help='The sample sizes to use (e.g. "[0, 3, 5]"')
    parser.add_argument('n_runs', help='The number of runs to use (e.g. 5)')
    parser.add_argument('samples_per_class', help='The number samples to use for validation per class (e.g. 250)')
    args = parser.parse_args()
    device = args.device
    model_name = args.model
    prompt_sample_sizes = eval(args.prompt_sample_sizes)
    n_runs = int(args.n_runs)
    samples_per_class = int(args.samples_per_class)

    if args.task == "out_of_order":
        train_df, val_df = get_pair_data(samples_per_class)
        input_att = "eventually_follows"
        task_prompt = general_task_prompt_order
        get_few_shot_prompt = get_few_shot_prompt_pairs
    elif args.task == "trace_anomaly":
        train_df, val_df = get_trace_data(samples_per_class)
        input_att = "trace"
        task_prompt = general_task_prompt
        get_few_shot_prompt = get_few_shot_prompt_traces
    else:
        raise ValueError("Task not recognized")

    model, tokenizer = get_model_and_tokenizer(model_name, device=device, quant_config=None)
    res_df = run_evaluation_loop(model_name=model_name, device=device, model=model, tokenizer=tokenizer,
                                 prompt_sample_sizes=prompt_sample_sizes, n_runs=n_runs, train_df=train_df,
                                 val_df=val_df, task_prompt=task_prompt,
                                 get_few_shot_prompt=get_few_shot_prompt, input_att=input_att)
    res_df.to_csv(EVAL_PATH /
                  (args.model.replace("/", "-")
                   + args.task + "_" + args.prompt_sample_sizes.replace("[", "").replace("]", "").replace(", ", "_")
                   + "_eval_results.csv"), index=False)


if __name__ == '__main__':
    main()
