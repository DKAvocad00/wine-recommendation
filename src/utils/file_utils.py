import pandas as pd


def read_csv(file_path, drop_duplicates=False):
    df = pd.read_csv(file_path, usecols=['points', 'title', 'description', 'price'], encoding='utf-8')
    df.dropna()
    if drop_duplicates:
        df = df.drop_duplicates(subset=['title'])
    return df


def read_file(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')

    return df

