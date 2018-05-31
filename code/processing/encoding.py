import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from utilities import normalize

import bisect

def one_hot(df, column, mapping=None):
    """ 
        Performs one-hot encoding.
        Example:
            Color: ['black', 'white', 'white']
            is encoded by
            Black: [1, 0, 0]
            White: [0, 1, 1]
            
        :param df: Data
        :param column: Column to encode
        :return: Encoded data
        :rtype: pd.Dataframe
    """
    x = df.copy()
    if mapping:
        x, mapping_ = label_encoding(x, column, mapping)
    else:
        x, mapping_ = label_encoding(x, column)

    x = pd.concat([x, pd.get_dummies(x[column], prefix=column)],axis=1)
    x.drop([column],axis=1, inplace=True)
    return x, mapping_

def likelihood(df, column, mapping=None):
    """ 
        Performs likelihood encoding.
            
        :param df: Data
        :param column: Column to encode
        :return: Encoded data
        :rtype: pd.Dataframe
    """
    x = df.copy()
    # Numerical columns.
    numericals = x.columns[x.dtypes != np.object]

    try: 
        pca = PCA()
        principal_axe = pca.fit(x[numericals].values).components_[0, :]
        # First principal component.
        pc1 = (principal_axe * x[numericals]).sum(axis=1)
    except:
        raise OSError('No numerical columns found, cannot apply likelihood encoding.')

    categories = x[column].unique()
    mapping_ = dict()
    for i, category in enumerate(categories):
        mapping_[category] = np.mean(pc1[x[column]==category])

    if mapping:
        if not mapping.keys() == mapping_.keys():
            for category in mapping_.keys() - mapping.keys():
                mapping_[category] = 0

    #x = x.replace({column:mapping_})
    x[column] = x[column].map(mapping_)
    return x, mapping_

def label(df, column, mapping=None):
    """ 
        Performs label encoding.
        Example:
            Color: ['blue', 'green', 'blue', 'pink']
            is encoded by
            Color: [1, 2, 1, 3]
            
        :param df: Data
        :param column: Column to encode
        :return: Encoded data
        :rtype: pd.Dataframe
    """
    x = df.copy()
    unique = x[column].unique()
    mapping_ = dict(zip(unique, np.arange(len(unique))))

    if mapping:
        if not mapping.keys() == mapping_.keys():
            diff = mapping_.keys() - mapping.keys()
            for category in diff:
                mapping_[category] = len(unique) - len(diff)
            for category in mapping.keys():
                mapping_[category] = mapping[category]

    #x = x.replace({column:mapping_})
    x[column] = x[column].map(mapping_)
    return x, mapping_
    

def frequency(columns, probability=False):
    """ /!\ Warning: Take only column(s) and not DataFrame /!\
        Frequency encoding:
            Pandas series to frequency/probability distribution.
        
        If there are several series, the outputs will have the same format.
        Example:
          C1: ['b', 'a', 'a', 'b', 'b']
          C2: ['b', 'b', 'b', 'c', 'b']
          
          f1: ['a': 2, 'b'; 3, 'c': 0]
          f2: ['a': 0, 'b'; 4, 'c': 1]
          
          Output: [[2, 3, 0], [0, 4, 1]] (with probability = False)
          
        :param probability: True for probablities, False for frequencies.
        :return: Frequency/probability distribution.
        :rtype: list
    """ # TODO error if several columns have the same header

    # If there is only one column, just return frequencies
    if not isinstance(columns[0], (list, np.ndarray, pd.Series)):
        return columns.value_counts(normalize=probability).values
    
    frequencies = []
    
    # Compute frequencies for each column
    for column in columns:
        f = dict()
        for e in column:
            if e in f:
                f[e] += 1
            else:
                f[e] = 1
        frequencies.append(f)
        
    # Add keys from other columns in every dictionaries with a frequency of 0
    # We want the same format
    for i, f in enumerate(frequencies):
        for k in f.keys():
            for other_f in frequencies[:i]+frequencies[i+1:]:
                if k not in other_f:
                    other_f[k] = 0
           
    # Convert to frequency/probability distribution
    res = []         
    for f in frequencies:
        l = list(f.values())
        if probability:
            # normalize between 0 and 1 with a sum of 1
            l = normalize(l)
        # Convert dict into a list of values
        res.append(l)
        # Every list will follow the same order because the dicts contain the same keys
                    
    return res
