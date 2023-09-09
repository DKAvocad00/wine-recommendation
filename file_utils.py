import pandas as pd


def read_file(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df.dropna()
    df = df.drop_duplicates(subset=['title'])

    return df
