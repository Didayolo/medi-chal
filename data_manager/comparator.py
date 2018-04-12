# Imports
from utilities import *
from scipy.stats import ttest_ind
from IPython.display import display
from norm import *
from sklearn.linear_model import LogisticRegression
import random

class Comparator():
    def __init__(self, ds1, ds2):
        """
            Constructor
            ds1 and ds2 are AutoML objects
            
        """
        # Datasets to compare
        self.ds1 = ds1
        self.ds2 = ds2
        
        # Check if ds1 and ds2 have the same features number
        assert (ds1.info['feat_num'] == ds2.info['feat_num']), "Datasets don't have the same features number, {} != {}".format(ds1.info['feat_num'], ds2.info['feat_num'])
        
        #Check if ds1 and ds2 are the exactly same dataset. Then no need to perform comparison.
        if self.ds1.get_data_as_df()['X'].equals(self.ds2.get_data_as_df()['X']):
            print("Datasets are equal")
        
        # Dictionary of distances between each descriptor of ds1 and ds2
        self.descriptors_dist = dict()
        self.compare_descriptors()
        
        # Features/metrics matrix
        self.comparison_matrix = pd.DataFrame(columns=ds1.get_data_as_df()['X'].columns.values)
        self.compute_comparison_matrix()

    def get_ds1(self):
        return self.ds1
        
    def get_ds2(self):
        return self.ds2

    def datasets_distance(self, axis=None, norm='manhattan'):
        """ Compute distance between ds1 and ds2
            Input:
              norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
        """
        data1 = self.ds1.get_processed_data()['X'].values
        data2 = self.ds2.get_processed_data()['X'].values
        return distance(data1, data2, axis=axis, norm=norm)

    def dcov(self):
        """ Compute the distance correlation between ds1 and ds2.
        """
        return distcorr(self.ds1.data['X'], self.ds2.data['X'])

    def t_test(self):
        """ Perform Student's t-test.
        """
        return ttest_ind(self.ds1.data['X'], self.ds2.data['X'])
         
    def compare_descriptors(self, norm='manhattan'):
        """ Compute distances between descriptors of ds1 and ds2.
            Input:
              norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
        """
        descriptors1 = self.ds1.descriptors
        descriptors2 = self.ds2.descriptors
        
        # For each descriptor
        for k in list(descriptors1.keys()):
            # Distance
            self.descriptors_dist[k] = distance(descriptors1[k], descriptors2[k], norm=norm)
            
    def compute_comparison_matrix(self):
        """ Compute a pandas DataFrame
            Columns: data features
            Rows: univariate comparison metrics (numerical or categorical)
        """
        
        data1 = self.ds1.get_processed_data()['X']
        data2 = self.ds2.get_processed_data()['X']
        
        columns = data1.columns.values
        for i, column in enumerate(columns):
        
            # Numerical
            if self.ds1.is_numerical[i] == 'numerical':
                self.comparison_matrix.at['Kolmogorov-Smirnov', column] = kolmogorov_smirnov(data1[column], data2[column])
            
            # Categorical, other
            else:
                f1 = to_frequency(data1[column])
                f2 = to_frequency(data2[column])
                
                self.comparison_matrix.at['Kullback-Leibler divergence', column] = kullback_leibler(f1, f2)
                self.comparison_matrix.at['Mutual information', column] = mutual_information(f1, f2)
                #self.comparison_matrix.at['Chi-square', column] = chi_square(f1, f2)
                
    def classify(self, clf=LogisticRegression()):
        """ Return the score (mean accuracy) of a classifier train on the data labeled with 0 or 1 according to their original dataset.
        """
        
        ds1 = self.ds1.get_processed_data()
        ds2 = self.ds2.get_processed_data()
    
        # Train set
        X1_train, X2_train = list(ds1['X_train'].values), list(ds2['X_train'].values)
        X_train = X1_train + X2_train
        y_train = [0] * len(X1_train) + [1] * len(X2_train)
        
        # Shuffle
        combined = list(zip(X_train, y_train))
        random.shuffle(combined)
        X_train[:], y_train[:] = zip(*combined)
        
        # Test set
        X1_test, X2_test = list(ds1['X_test'].values), list(ds2['X_test'].values)
        X_test = X1_test + X2_test
        y_test = [0] * len(X1_test) + [1] * len(X2_test)
        
        # Training
        clf.fit(X_train, y_train)
        
        # Score
        return clf.score(X_test, y_test)
        
    def show_classifier_score(self, clf=LogisticRegression()):
        """ Display classify method result
        """
        score = self.classify(clf=clf).round(5)
        print(clf)
        print('\n')
        printmd('** Score: **' + str(score))
        print('\n')
          
    def show_descriptors(self):
        """ Show descriptors distances between ds1 and ds2
        """
        for k in list(self.descriptors_dist.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.descriptors_dist[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value))

    def show_comparison_matrix(self):
        """ Display inter-columns comparison
        """
        display(self.comparison_matrix)
