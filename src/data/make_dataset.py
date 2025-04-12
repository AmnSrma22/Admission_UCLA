import pandas as pd

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.drop(columns=['Serial_No'], errors='ignore', inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df