import numpy as np
import pandas as pd
from encoding import *

def get_types(df):
    x = df.copy()
    dtypes = list()
    for column in x.columns:
        n = len(x[column].unique())
        try:
            sum = x[column].sum()
        except:
            sum = -np.inf
        if n == 2:
            dtypes.append('Binary')
        elif (n > 2 and (sum == n*(n-1)/2 or sum == n*(n+1)/2)) or any(isinstance(i, str) for i in x[column]):
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
        cols_to_remove = np.where(np.array(types)=='Categorical')[0]
        x = x.drop(x.columns[cols_to_remove], axis=1)
        types = np.delete(types, cols_to_remove)

    for column in x.columns[[i for i, j in enumerate(types) if j=='Numerical']].values:
        x[column] = x[column].fillna(x[column].median())
        # Replace +Inf by the maximum and -Inf by the minimum.
        x[column] = x[column].replace(np.inf, x[column].max())
        x[column] = x[column].replace(-np.inf, x[column].min())
        if normalization == 'standard':
            x[column] = (x[column] - x[column].mean()) / x[column].std()
        elif normalization == 'min-max':
            x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())

    for column in x.columns[[i for i, j in enumerate(types) if j=='Binary']].values:
        x = label_encoding(x, column)

    for column in x.columns[[i for i, j in enumerate(types) if j=='Categorical']].values:
        # Replace NaN with 'missing'.
        x[column] = x[column].fillna('missing')
        if categorical=='one-hot':
            x = one_hot_encoding(x, column)
        elif categorical=='likelihood':
            x = likelihood_encoding(x, column)
        elif categorical=='label':
            x = label_encoding(x, column)

    return x