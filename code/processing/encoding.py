import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def one_hot_encoding(df, column):
    df = label_encoding(df, column)
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)],axis=1)
    df.drop([column],axis=1, inplace=True)
    return df

def likelihood_encoding(df, column):
    pca = PCA().fit_transform(df[df.columns[df.dtypes != np.object]].values)
    pc = pca[:, 0]
    categories = df[column].unique()
    for i, category in enumerate(categories):
        df[column] = df[column].replace(category, 1 / np.mean(pc[df[column]==category]))
    return df

def label_encoding(df, column):
    df[column] = LabelEncoder().fit_transform(df[column])
    return df