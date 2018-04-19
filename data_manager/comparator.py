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
        
        # Metrics and plots for privacy and resemblance
        # TODO
        self.mda1 = None
        self.mda2 = None

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
            if k in descriptors2.keys():
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
                self.comparison_matrix.at['Jensen-Shannon divergence', column] = jensen_shannon(f1, f2)
                #self.comparison_matrix.at['Chi-square', column] = chi_square(f1, f2)
                
    def classify(self, clf=LogisticRegression()):
        """ Return the score (mean accuracy) of a classifier train on the data labeled with 0 or 1 according to their original dataset.
            Input:
              clf: the classifier. It has to have fit(X,y) and score(X,y) methods.
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
        """ Display the score (mean accuracy) of a classifier train on the data labeled with 0 or 1 according to their original dataset.
            (return of 'classify' method)
            Input:
              clf: the classifier. It has to have fit(X,y) and score(X,y) methods.
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


    def compute_mda(self, norm='manhattan', precision=0.2, threshold=None, area='simpson'):
        """ Compute the accumulation of minimum distances from one dataset to other.
            Use for privacy/resemblance metrics
        """
        # Distributions
        A = self.ds1.get_processed_data()['X'].as_matrix()
        B = self.ds2.get_processed_data()['X'].as_matrix()
        
        # Distances to nearest neighbors
        mdA, mdB = minimum_distance(A, B, norm=norm)
        
        # Curve and metrics
        self.mda1 = compute_mda(mdA, precision=precision, threshold=threshold, area=area)
        self.mda2 = compute_mda(mdB, precision=precision, threshold=threshold, area=area)
        
    
    def show_mda(self):
        """ Show the accumulation of minimum distances from one dataset to other.
            Use for privacy/resemblance metrics
        """
        if self.mda1 == None:
            self.compute_mda()
            
        (xA, yA), (privacyA, resemblanceA), thresholdA = self.mda1
        (xB, yB), (privacyB, resemblanceB), thresholdB = self.mda2
        
        # Plot A
        print('DS1')
        plt.plot(xA, yA)
        plt.axvline(x=thresholdA, color='r', label='threshold')
        plt.xlabel('Distance d')
        plt.ylabel('Number of minimum distance < d')
        plt.title('MDA ds1 to ds2')
        plt.show()
        
        printmd('** Privacy: **' + str(privacyA))
        printmd('** Resemblance: **' + str(resemblanceA))
        
        # Plot B
        print('DS2')
        plt.plot(xB, yB)
        plt.axvline(x=thresholdB, color='r', label='threshold')
        plt.xlabel('Distance d')
        plt.ylabel('Number of minimum distance < d')
        plt.title('MDA ds2 to ds1')
        plt.show()
        
        printmd('** Privacy:** ' + str(privacyB))
        printmd('** Resemblance:** ' + str(resemblanceB))
        
     
    def show_mmd(self):
        """ Compute and show MMD between ds1 and ds2
        """
        A = self.ds1.get_processed_data()['X'].as_matrix()
        B = self.ds2.get_processed_data()['X'].as_matrix()
        score = mmd(A, B)
        print('Maximum mean discrepancy: ' + str(score))
