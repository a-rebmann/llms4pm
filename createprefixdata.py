import pandas as pd

from const import DATA_ROOT


def get_prefixes(x):
    tr = x["trace"]
    return [tr[:i] for i in range(1, len(tr))] +[tr], [tr[i] for i in range(1, len(tr))] +  ['[END]']


def generate_prefix_data(df):
    df[["prefixes", "next"]] = df.apply(lambda x: pd.Series([*get_prefixes(x)]), axis=1)
    return df


def expand_df_regular_traces(df):
    per_trace_records = []
    for i, row in df.iterrows():
        traces = eval(row['string_traces'])
        labels = eval(row['labels'])
        for tix, trace in enumerate(traces):
            if len(trace) < 2:
                continue
            per_trace_records.append({
                'model_id': row['model_id'],
                'revision_id': row['revision_id'],
                'trace': traces[tix],
                'unique_activities': row['unique_activities']
            })

    df = pd.DataFrame(per_trace_records)
    return df


def expand_trace_df(df):
    per_trace_records = []
    for i, row in df.iterrows():
        for prefix, next_ in zip(row["prefixes"], row['next']):
            per_trace_records.append({
                'model_id': row['model_id'],
                'revision_id': row['revision_id'],
                'trace': row['trace'],
                'prefix': prefix,
                'next': next_,
                'unique_activities': row['unique_activities']
            })
    df = pd.DataFrame(per_trace_records)
    return df


if __name__ == "__main__":
    model_df = pd.read_csv(DATA_ROOT / "eval_data_ooo.csv")
    trace_df = expand_df_regular_traces(model_df)
    trace_df = generate_prefix_data(trace_df)
    prefix_df = expand_trace_df(trace_df)
    # pickle the data
    prefix_df.to_pickle(DATA_ROOT + "prefix_data.pkl")

