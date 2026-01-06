import pandas as pd

def load_cmaps_data(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df.drop([26, 27], axis=1, inplace=True)
    df.columns = ["unit", "time", "op1", "op2", "op3"] + \
                 [f"sensor_{i}" for i in range(1, 22)]
    return df
