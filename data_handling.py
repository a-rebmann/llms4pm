from typing import Any, cast

import torch
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import pandas as pd
from sklearn.model_selection import train_test_split

import pickle


def load_pkl(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def remove_duplicates(pair_df):
    columns = ["revision_id", "model_id", "unique_activities"]
    if "eventually_follows" in pair_df.columns:
        columns.append("eventually_follows")
    pair_df = pair_df.drop_duplicates(subset=columns)
    return pair_df


def setify(x: str):
    set_: set[str] = eval(x)
    assert isinstance(set_, set), f"Conversion failed for {x}"
    return set_


EVAL_PATH = "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/"


def deduplicate_traces(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["model_id", "revision_id", "trace"])


def split_by_model(df, split_sizes: list[float] = [0.2, 0.1], random_state=4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["id"] = df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    model_ids = df["id"].unique()
    train_val_ids, test_ids = train_test_split(
        model_ids, test_size=split_sizes[-1], random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=split_sizes[-2], random_state=random_state
    )
    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]
    return train_df, val_df, test_df


def load_pairs(split: str = "train") -> Dataset:
    eval_pairs = load_pkl(
        "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/eval_train_data_pairs.pkl"
    )
    eval_pairs["labels"] = ~(eval_pairs.out_of_order)
    eval_pairs = remove_duplicates(eval_pairs)
    eval_pairs.trace = eval_pairs.trace.apply(lambda x: tuple(x))
    eval_pairs.unique_activities = eval_pairs.unique_activities.apply(setify)
    columns = [
        "model_id",
        "revision_id",
        "unique_activities",
        "labels",
        "eventually_follows",
    ]
    eval_pairs = eval_pairs.loc[:, columns]
    train, val, test = split_by_model(eval_pairs)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)


def load_split(split: str = "train") -> Dataset:
    eval_train: pd.DataFrame = load_pkl(
        "/work/ws/ma_fabiasch-tx/trident-dialect-copa/data/eval_train_data_traces.pkl"
    )
    eval_train["labels"] = ~(eval_train.apply(lambda x: len(x["label"]) > 0, axis=1))
    eval_train = remove_duplicates(eval_train)
    eval_train.trace = eval_train.trace.apply(lambda x: tuple(x))
    eval_train.unique_activities = eval_train.unique_activities.apply(setify)
    columns = ["model_id", "revision_id", "unique_activities", "trace", "labels"]
    eval_train = eval_train.loc[:, columns]
    train, val, test = split_by_model(eval_train)
    if split == "train":
        return Dataset.from_pandas(train)
    elif split == "val":
        return Dataset.from_pandas(val)
    else:
        return Dataset.from_pandas(test)


def preprocess(examples: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, trace_ in zip(
        examples["unique_activities"], examples["trace"]
    ):
        input_ = f"Set of process activities: {str(unique_activities_)}\nProcess: {str(list(trace_))}\n: Valid: "
        inputs.append(input_)
    batch = tokenizer(inputs)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_pairs(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # eventually_follows: list[tuple[str, str]]
    # labels: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, activities_ in zip(
        examples["unique_activities"], examples["eventually_follows"]
    ):
        input_ = f"Set of activities: {str(unique_activities_)}\nOrdered activities: {str(list(activities_))}\n: Valid: "
        inputs.append(input_)
    batch = tokenizer(inputs)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_seq_clf(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # trace: list[tuple[str]]
    # label: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, trace_ in zip(
        examples["unique_activities"], examples["trace"]
    ):
        input_ = f"Set of process activities: {str(unique_activities_)}\nProcess: {str(list(trace_))}"
        inputs.append(input_)
    batch = tokenizer(inputs, max_length=512, truncation=True)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch


def preprocess_pairs_seq_clf(
    examples: dict, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    # eventually_follows: list[tuple[str, str]]
    # labels: list[bool]
    # unique_activities: list[set[str]]
    inputs = []
    for unique_activities_, activities_ in zip(
        examples["unique_activities"], examples["eventually_follows"]
    ):
        input_ = f"Set of activities: {str(unique_activities_)}\nOrdered activities: {str(list(activities_))}"
        inputs.append(input_)
    batch = tokenizer(inputs, truncation=True)
    batch["labels"] = torch.LongTensor([int(label) for label in examples["labels"]])
    return batch