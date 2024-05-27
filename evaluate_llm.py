import pickle
import warnings
import os
import argparse
from typing import cast

from torch import nn
from tqdm import tqdm
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer, MixtralForCausalLM,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from const import MISTRAL_MODEL, EVAL_PATH, TASKS, DATA_ROOT
from llm.prompts import general_task_prompt_order, get_few_shot_prompt_pairs, general_task_prompt, \
    get_few_shot_prompt_traces, get_few_shot_prompt_prefix, next_activity_prompt

warnings.filterwarnings("ignore")
tqdm.pandas()


def split_by_model(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    with open(
            DATA_ROOT / "train_val_test.pkl", "rb"
    ) as file:
        train_ids, val_ids, test_ids = pickle.load(file)
    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]
    return train_df, val_df, test_df


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


def get_labelled_tokens(tokenizer, label_tokens: list[str]) -> list[int]:
    """Returns the indices of label tokens in tokenizer vocabulary."""
    out = []
    for token in label_tokens:
        input_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        assert len(input_ids) == 1, f"{token} is of length {len(input_ids)}"
        out.append(input_ids[0])
    return out


def generate_binary_output(model_name, device, model, tokenizer, prompt):
    if model_name == MISTRAL_MODEL:
        prompt = "[INST]" + prompt + "[/INST]"
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
    )
    first_token_probs = outputs.scores[0][0]
    true = tokenizer("True")["input_ids"][1]
    false = tokenizer("False")["input_ids"][1]
    # option_scores = (
    #     first_token_probs[[true, false]].float().cpu().numpy()
    # )  # True, False
    # print(option_scores)
    # pred = np.array(["True", "False"])[np.argsort(option_scores)[::-1][:1]]
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    if "True" in decoded[0]:
        return True
    elif "False" in decoded[0]:
        return False
    print(decoded[0], "neither True nor False")
    print(tokenizer.decode(generated_tokens[0][0]))
    return False  # tokenizer.decode(generated_tokens[0][0])


def generate_activity_output(model_name, device, model, tokenizer, prompt, activities):
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
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for activity in alphabet:  # + ["[END]"]:
        if activity in decoded[0]:
            print(activity, "found")
            ix = alphabet.index(activity)
            if len(activities) <= ix:
                print("Activity not in list")
                return "[END]"
            return activities[ix]
    print(decoded[0], "does not contain any activity")
    return "[END]"


def _get_act_list(list_of_activities):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    act_list = "0. [END]\n"
    for idx, act in enumerate(list_of_activities):
        act_list += f"{alphabet[idx]}. {act}\n"
    return act_list


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
            if input_att == "trace":
                val_df["y"] = val_df.progress_apply(
                    lambda x: generate_binary_output(model_name, device, model, tokenizer,
                                                     this_prompt +
                                                     str(x["unique_activities"]) + "\n"
                                                     + "Trace:" + str(
                                                         x[input_att]) + "\nValid:"),
                    axis=1)
            elif input_att == "eventually_follows":
                val_df["y"] = val_df.progress_apply(
                    lambda x: generate_binary_output(model_name, device, model, tokenizer,
                                                     this_prompt +
                                                     str(x["unique_activities"]) + "\n"
                                                     + "1. Activity:" + str(x[input_att][0]) + "\n"
                                                     + "2. Activity:" + str(
                                                         x[input_att][1]) + "\nValid:"),
                    axis=1)
            elif input_att == "next":
                val_df["y"] = val_df.progress_apply(
                    lambda x: generate_activity_output(model_name, device, model, tokenizer,
                                                       this_prompt +
                                                       _get_act_list(x["unique_activities"]) + "\n"
                                                       + "Sequence of activities:" + str(
                                                           x["prefix"]) + "\nAnswer:",
                                                       activities=list(x["unique_activities"])),
                    axis=1)
            print("Detected raw", val_df['y'].value_counts())
            # val_df['y'] = val_df['y'].apply(lambda x: True if x == 'True' else False)
            # print("Detected parsed", val_df['y'].value_counts())
            if input_att == "trace":
                true_labels = ~val_df['anomalous']
            elif input_att == "eventually_follows":
                true_labels = ~val_df['out_of_order']
            elif input_att == "next":
                true_labels = val_df['next']
            else:
                raise NotImplemented
            print("True parsed", true_labels.value_counts())
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
            # add per class values
            if input_att != "next":
                # compute precision, recall, f1, support per class
                precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels)
                for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
                    rec[f"precision_{i}"] = p
                    rec[f"recall_{i}"] = r
                    rec[f"f1_{i}"] = f
                    rec[f"support_{i}"] = s
            print(rec)
            result_records.append(rec)
            done = True
    df = pd.DataFrame(result_records)
    return df


bnb_config = (BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
))


def get_pair_data(samples_per_class, with_trace=False):
    pair_df = pd.read_pickle(EVAL_PATH / "eval_train_data_pairs_balanced.pkl")
    pair_df["unique_activities"] = pair_df["unique_activities"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    if not with_trace:
        pair_df = pair_df.drop_duplicates(subset=["revision_id", "model_id", "eventually_follows"])
    # Split
    pair_train_df, pair_val_df, pair_test_df = split_by_model(pair_df)

    pair_val_df_positive = pair_val_df[~pair_val_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
    pair_val_df_negative = pair_val_df[pair_val_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
    pair_val_df = pd.concat([pair_val_df_negative, pair_val_df_positive])
    # shuffle the data
    pair_val_df = pair_val_df.sample(frac=1, random_state=4).reset_index(drop=True)
    # sample data to have equal number of positive and negative samples test
    if samples_per_class*2 < len(pair_test_df):
        pair_test_df_positive = pair_test_df[~pair_test_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
        pair_test_df_negative = pair_test_df[pair_test_df["out_of_order"]].sample(n=samples_per_class, random_state=4)
        pair_test_df = pd.concat([pair_test_df_negative, pair_test_df_positive])
    # shuffle the data
    pair_test_df = pair_test_df.sample(frac=1, random_state=4).reset_index(drop=True)
    return pair_train_df, pair_val_df, pair_test_df


def get_trace_data(samples_per_class):
    trace_df = pd.read_pickle(EVAL_PATH / "eval_train_data_traces_balanced.pkl")
    trace_df["unique_activities"] = trace_df["unique_activities"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    trace_df["anomalous"] = trace_df.progress_apply(lambda x: len(x["label"]) > 0, axis=1)
    # Split
    trace_train_df, trace_val_df, trace_test_df = split_by_model(trace_df)
    # Sample data to have equal number of positive and negative samples
    trace_val_df_positive = trace_val_df[~trace_val_df["anomalous"]].sample(n=samples_per_class, random_state=4)
    trace_val_df_negative = trace_val_df[trace_val_df["anomalous"]].sample(n=samples_per_class, random_state=4)
    trace_val_df = pd.concat([trace_val_df_positive, trace_val_df_negative])
    # shuffle the data
    trace_val_df = trace_val_df.sample(frac=1, random_state=4).reset_index(drop=True)
    # Sample data to have equal number of positive and negative samples
    if samples_per_class*2 < len(trace_test_df):
        trace_test_df_positive = trace_test_df[~trace_test_df["anomalous"]].sample(n=samples_per_class, random_state=4)
        trace_test_df_negative = trace_test_df[trace_test_df["anomalous"]].sample(n=samples_per_class, random_state=4)
        trace_test_df = pd.concat([trace_test_df_positive, trace_test_df_negative])
    # shuffle the data
    trace_test_df = trace_test_df.sample(frac=1, random_state=4).reset_index(drop=True)
    return trace_train_df, trace_val_df, trace_test_df


def get_prefix_data(samples_per_class):
    prefix_df = pd.read_pickle(EVAL_PATH / "eval_train_prefix_data.pkl")
    prefix_df["unique_activities"] = prefix_df["unique_activities"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    prefix_df = prefix_df[~prefix_df["next"].str.contains("END")]
    prefix_df["prefix"] = prefix_df["prefix"].apply(lambda x: tuple(x))
    prefix_df["unique_activities"] = prefix_df["unique_activities"].apply(lambda x: tuple(x))
    prefix_df = prefix_df.drop_duplicates(subset=["revision_id", "model_id", "prefix", "unique_activities"])
    prefix_df["prefix"] = prefix_df["prefix"].apply(lambda x: list(x))
    prefix_df["unique_activities"] = prefix_df["unique_activities"].apply(lambda x: set(x))
    prefix_train_df, prefix_val_df, prefix_test_df = split_by_model(prefix_df)
    prefix_val_df = prefix_val_df.sample(n=samples_per_class * 2, random_state=4)
    prefix_test_df = prefix_test_df.sample(n=samples_per_class * 2, random_state=4)
    return prefix_train_df, prefix_val_df, prefix_test_df


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
        train_df, val_df, test_df = get_pair_data(samples_per_class)
        input_att = "eventually_follows"
        task_prompt = general_task_prompt_order
        get_few_shot_prompt = get_few_shot_prompt_pairs
    elif args.task == "trace_anomaly":
        train_df, val_df, test_df = get_trace_data(samples_per_class)
        input_att = "trace"
        task_prompt = general_task_prompt
        get_few_shot_prompt = get_few_shot_prompt_traces
    elif args.task == "next_activity":
        print("next activity prediction. samples per class will be ignored")
        train_df, val_df, test_df = get_prefix_data(samples_per_class)
        input_att = "next"
        task_prompt = next_activity_prompt
        get_few_shot_prompt = get_few_shot_prompt_prefix
    else:
        raise ValueError("Task not recognized")

    model, tokenizer = get_model_and_tokenizer(model_name, device=device, quant_config=None)
    print(len(train_df), len(val_df), len(test_df))

    #model = cast(MixtralForCausalLM, model)
    #model.lm_head = torch.nn.Identity()
    #embeddings = model.get_output_embeddings()
    #ids = get_labelled_tokens(tokenizer, ["True", "False"] if input_att != "next" else list(
    #    "0ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    # index token ids of output embedding matrix
    #_clf_head = nn.Parameter(embeddings.weight[ids].clone().T)
    #_clf_head.requires_grad = False

    res_df = run_evaluation_loop(model_name=model_name, device=device, model=model, tokenizer=tokenizer,
                                 prompt_sample_sizes=prompt_sample_sizes, n_runs=n_runs, train_df=train_df,
                                 val_df=test_df, task_prompt=task_prompt,
                                 get_few_shot_prompt=get_few_shot_prompt,
                                 input_att=input_att)
    res_df.to_csv(EVAL_PATH /
                  (args.model.replace("/", "-")
                   + args.task + "_" + args.prompt_sample_sizes.replace("[", "").replace("]", "").replace(", ", "_")
                   + "_samples_per_class" + str(samples_per_class) + "_runs" + str(n_runs) + str(
                              general_task_prompt_order[-10:])
                   + "_eval_results.csv"), index=False)


if __name__ == '__main__':
    main()
