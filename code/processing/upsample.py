import sys
import pandas as pd
import numpy as np


def read_data(fname):
    """read in the data"""
    return pd.read_csv(fname)


def balance_class(df, col, n):
    """balance `col` in dataframe `df`"""
    class0 = np.where(df[col] == 0)[0]
    class1 = np.where(df[col] == 1)[0]

    # sample evenly
    class0_sampled = np.random.choice(class0, size=n//2, replace=True)
    class1_sampled = np.random.choice(class1, size=n//2, replace=True)

    # put back together
    df = pd.concat((df.iloc[class0_sampled], df.iloc[class1_sampled]))
    df = df.reset_index()
    return df


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: python {sys.argv[0]} <filename> <column_to_balance>')
        sys.exit()

    filename = sys.argv[1]
    data = read_data(filename)

    # balance the classes with size 10 * orig_n
    data = balance_class(data, sys.argv[2], data.shape[0] * 10)

    data.to_csv(f'{filename.strip(".csv")}_upsampled.csv', index=False)