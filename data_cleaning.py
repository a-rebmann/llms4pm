
import pandas  as pd
from const import EVAL_PATH

def clean_all():
    # for all csvs in the eval path clean the labels in all columns consistently
    for csv in EVAL_PATH.glob('*.csv'):
        clean_csv(csv)

def clean_csv(csv):
    # read with pandas
    df = pd.read_csv(csv)
    print(f'Cleaning {csv}')
    # iterate over all columns
    for col in df.columns:
        # clean the column
        clean_column(df, col)

def clean_column(df, col_name):
   df[col_name] = df[col_name].apply(lambda x: eval(x) if isinstance(x, str) else x)

if __name__ == '__main__':
    clean_all()