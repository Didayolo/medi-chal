import numpy as np
import pandas as pd
from encoding import *

def get_types(df):
    """ Get variables types: Numeric, Binary or Categorical.
    
        :param df: pandas DataFrame
        :return: List of type of each variable
        :rtype: list
    """
    
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

def processing(df, normalization='mean', encoding='label', missing='median'):
    """
        Return preprocessed DataFrame
        
        :param df: pandas DataFrame
        :param encoding: ['one-hot', 'likelihood', 'label']
        :param normalization: ['standard', 'minmax', None]
        :return: Preprocessed data
        :rtype: pd.DataFrame
    """
    
    x = df.copy()
    types = get_types(x)
    
    if encoding in ['None', 'none'] or encoding is None:
        cols_to_remove = np.where(np.array(types)=='Categorical')[0]
        x = x.drop(x.columns[cols_to_remove], axis=1)
        types = np.delete(types, cols_to_remove)

    # For numerical variables
    for column in x.columns[[i for i, j in enumerate(types) if j=='Numerical']].values:
        
        # Replace NaN with the median of the variable value
        if missing == 'remove':
            x = x.dropna(axis=0)
        if missing == 'median':
            x[column] = x[column].fillna(x[column].median())
        if missing == 'mean':
            x[column] = x[column].fillna(x[column].mean())
        
        # Replace +Inf by the maximum and -Inf by the minimum.
        x[column] = x[column].replace(np.inf, x[column].max())
        x[column] = x[column].replace(-np.inf, x[column].min())
        
        # Mean normalization
        if normalization == 'standard':
            x[column] = (x[column] - x[column].mean()) / x[column].std()
        
        # Min-max normalization
        elif normalization == 'min-max':
            x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min())

    # For binary variables
    for column in x.columns[[i for i, j in enumerate(types) if j=='Binary']].values:
        x = label_encoding(x, column)

    # For categorigal variables
    for column in x.columns[[i for i, j in enumerate(types) if j=='Categorical']].values:

        # Replace NaN with 'missing'.
        x[column] = x[column].fillna('missing')
        
        # One-hot encoding: [0, 0, 1]
        if encoding=='one-hot':
            x = one_hot_encoding(x, column)
            
        # Likelihood encoding
        elif encoding=='likelihood':
            x = likelihood_encoding(x, column)
            
        # Label encoding: [1, 2, 3]
        elif encoding=='label':
            x = label_encoding(x, column)

    return x
