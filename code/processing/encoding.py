import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

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
        pca = pca.fit_transform(x[numericals].values)
        pc1 = pca[:, 0]
    except:
        raise OSError('No numerical columns found, cannot apply likelihood encoding.')

    categories = x[column].unique()
    mapping_ = dict()
    for i, category in enumerate(categories):
        mapping_[category] = 1 / np.mean(pc1[x[column]==category])

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
