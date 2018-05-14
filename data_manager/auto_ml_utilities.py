import numpy as np
import pandas as pd

def get_types(df):
    x = df.copy()
    dtypes = list()
    for column in x.columns:
        n = len(np.unique(x[column]))
        if n == 2:
            dtypes.append('Binary')
        elif (n > 2) and (
            x[column].sum() == n*(n-1)/2 or 
            x[column].sum() == n*(n+1)/2 or
            type(x[column][0]) == np.str):
            dtypes.append('Categorical')
        else:
            dtypes.append('Numerical')
    return dtypes

def preprocess(df, types, normalization, encoding):
    x = df.copy()
    if encoding=='none':
        cols_to_keep = np.where(types=='Categorical')[0]
        x = x[:, cols_to_keep]
    else:
        for i, column in enumerate(x.columns):
            if types[i] == 'Binary':
                x = label_encoding(x, column)
            elif types[i] == 'Categorical':
                if encoding=='one-hot':
                    x = one_hot_encoding(x, column)
                elif encoding=='likelihood':
                    x = likelihood_encoding(x, column)
                elif encoding=='label':
                    x = label_encoding(x, column)
            elif types[i] == 'Numeric':
                if normalization == 'standard':
                    x[column] = (x[column] - x[column].mean()) / x[column].std()
    return x.values