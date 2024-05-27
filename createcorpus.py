import pandas as pd
from langdetect import LangDetectException, detect, DetectorFactory

import data.model_parser as parser
from const import DATA_ROOT
from data.model_to_log import Model2LogConverter
import pm4py
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.WARN)


def detect_language(x):
    try:
        res = detect(str(x).replace("{", "").replace("}", ""))
    except LangDetectException:
        print(f"Language detection failed for {x}")
        res = "unknown"
    return res


# Set a seed for reproducibility (optional)
DetectorFactory.seed = 0


def run():
    csv_paths = parser.get_csv_paths()
    for csv_path in csv_paths:
        model_df = parser.parse_models([csv_path])
        print(f"Total models in this frame {len(model_df)}")
        conv = Model2LogConverter()
        model_df = conv.convert_models_to_pn_df(model_df)
        model_df = conv.check_soundness(model_df)
        model_df = model_df[model_df["sound"]]
        model_df = conv.create_log(model_df)
        model_df = model_df[model_df["traces"].apply(lambda x: x is not None and len(x) > 0 and len(x[0]) > 0)]
        model_df["string_traces"] = model_df["traces"].apply(lambda x: [[e["concept:name"] for e in t] for t in x])
        model_df["dfg"] = model_df["traces"].apply(lambda x: create_dfg(x))
        model_df["pt"] = model_df.apply(lambda x: create_pt(x["pn"][0], x["pn"][1], x["pn"][2]), axis=1)
        model_df = model_df[model_df["pt"].apply(lambda x: x is not None)]
        print(f"Models for which a sound WF net, PT, DFG, and traces could be generated {len(model_df)}")
        # only keep model id, revision id, string traces and dfg
        model_df = model_df[["model_id", "revision_id", "name", "string_traces", "dfg", "pn", "pt"]]
        model_df["unique_activities"] = model_df["string_traces"].apply(lambda x: set([e for t in x for e in t]))
        # detect model language
        model_df["language"] = model_df["unique_activities"].apply(lambda x: detect_language(x))
        model_df = model_df[model_df["language"] == "en"]
        model_df = model_df.drop_duplicates(subset=['unique_activities'])
        # save to the same Path but with a different name
        model_df.to_csv(str(csv_path).replace(".csv", "_corpus.csv"), index=False)


def create_dfg(log):
    dfg, _, _ = pm4py.discover_dfg(log, case_id_key='case:concept:name', activity_key='concept:name',
                                   timestamp_key='time:timestamp')
    return list(dfg.keys())


def create_pt(pn, im, fm):
    try:
        pt = pm4py.convert_to_process_tree(pn, im, fm)
    except Exception as e:
        print(f"Error during process tree creation: {e}")
        pt = None
    return pt


def update_corpus():
    c_path = DATA_ROOT / "basic_cleaned_corpus.csv"
    model_df = pd.read_csv(c_path)
    c_path = DATA_ROOT / "basic_cleaned_corpus_.csv"
    model_df["string_traces"] = model_df["string_traces"].apply(eval)
    model_df["unique_activities"] = model_df["string_traces"].apply(lambda x: set([e for t in x for e in t]))
    model_df = model_df[model_df["unique_activities"] > 1]
    # detect model language
    model_df["language"] = model_df["unique_activities"].apply(lambda x: detect_language(x))
    model_df = model_df[model_df["language"] == "en"]
    model_df = model_df.drop_duplicates(subset=['unique_activities'])
    # save to the same Path but with a different name
    model_df.to_csv(c_path, index=False)


if __name__ == "__main__":
    run()
    #update_corpus()
