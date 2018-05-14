import numpy as np
import pandas as pd
from encoding import *

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

def preprocess(df, normalization, categorical):
    """
        df = Pandas DataFrame
        normalization = ['standard', 'min-max']
        categorical = ['one-hot', 'likelihood', 'label']
    """
    x = df.copy()
    types = get_types(x)
    if categorical=='none':
        cols_to_keep = np.where(types=='Categorical')[0]
        x = x[:, cols_to_keep]
    else:
        for i, column in enumerate(x.columns):
            if types[i] == 'Binary':
                x = label_encoding(x, column)
            elif types[i] == 'Categorical':
                # Replace NaN with 'missing'.
                x[column] = x[column].fillna('missing')
                if categorical=='one-hot':
                    x = one_hot_encoding(x, column)
                elif categorical=='likelihood':
                    x = likelihood_encoding(x, column)
                elif categorical=='label':
                    x = label_encoding(x, column)
            elif types[i] == 'Numeric':
                # Replace NaN with median.
                x[column] = x[column].fillna(x[column].median())
                # Replace +Inf by the maximum and -Inf by the minimum.
                x[column] = x[column].replace(np.inf, max(x[column]))
                x[column] = x[column].replace(-np.inf, min(x[column]))
                if normalization == 'standard':
                    x[column] = (x[column] - x[column].mean()) / x[column].std()
                elif normalization == 'min-max':
                    x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())
    return x