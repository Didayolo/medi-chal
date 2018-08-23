# Imports

import sys
from utilities import *
from scipy.stats import ttest_ind
from IPython.display import display
from metric import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random
from encoding import frequency
from auto_ml import AutoML
import matplotlib.pyplot as plt


class Comparator():
    def __init__(self, ds1, ds2=None, test_size=0.2):
        """
            Constructor
            
            :param ds1: AutoML object representing the first dataset.
            :param ds2: AutoML object representing the second dataset.
                        If ds2 is None, the comparator will compare the train and test sets of ds1.
            :param test_size: Test set size for the auto-comparator case.
        """
        # Datasets to compare
        if ds2 is None:
            print('1 dataset detected: comparison between train and test sets.')
            X_train = ds1.get_data('X_train')
            X_test = ds1.get_data('X_test')
            y_train = None
            y_test = None
            
            if 'y' in ds1.subsets:
                y_train = ds1.get_data('y_train')
                y_test = ds1.get_data('y_test')
            
            self.ds1 = AutoML.from_df('tmp', 'tmp_train', X_train, y=y_train, test_size=test_size)
            self.ds2 = AutoML.from_df('tmp', 'tmp_test', X_test, y=y_test, test_size=test_size)
        
        else:
            print('2 datasets detected: ready for comparison.')
            self.ds1 = ds1
            self.ds2 = ds2
        
        # Processing
        #self.process_data()
        
        # Check if ds1 and ds2 have the same features number
        if self.ds1.info['feat_num'] != self.ds2.info['feat_num']:
            print("WARNING: Datasets don't have the same features number, {} != {}".format(self.ds1.info['feat_num'], self.ds2.info['feat_num']))
        
        #Check if ds1 and ds2 are the exactly same dataset. Then no need to perform comparison.
        if self.ds1.get_data().equals(self.ds2.get_data()):
            print("Datasets are equal")
        
        # Dictionary of distances between each descriptor of ds1 and ds2
        self.descriptors_dist = dict()
        self.compute_descriptors()
        
        # Features/metrics matrix
        self.comparison_matrix = None
        
        # Metrics and plots for privacy and resemblance
        # TODO
        self.mda = None
        #self.mda2 = None

    def get_ds1(self):
        return self.ds1
        
    def get_ds2(self):
        return self.ds2
        
    def process_data(self, **kwargs):
        """ Apply process_data method on ds1 and ds2
        """
        self.ds1.process_data(**kwargs)
        self.ds2.process_data(**kwargs)

    def datasets_distance(self, axis=None, norm='manhattan'):
        """ Compute distance between ds1 and ds2
            
            :param norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
        """
        #self.process_data()
        A = self.ds1.get_data('X', processed=False).values
        B = self.ds2.get_data('X', processed=False).values
        #print(A)
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return distance(A, B, axis=axis, norm=norm)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A[np.random.choice(A.shape[0], min_sample_num, replace=False)]
                B1 = B[np.random.choice(B.shape[0], min_sample_num, replace=False)]
                #print(A1)
                #print(B1)
                return distance(A1, B1, axis=axis, norm=norm)

    def distance_correlation(self):
        #self.process_data()
        
        A = self.ds1.get_data('X', processed = False)
        B = self.ds2.get_data('X', processed = False)
        
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return distance_correlation(A, B)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A.sample(frac = min_sample_num/ds1_shape[0], replace = False)
                B1 = B.sample(frac = min_sample_num/ds2_shape[0], replace = False)
                return distance_correlation(A1, B1)
        
    def dcorr(self):
        #self.process_data()
        
        A = self.ds1.get_data('X', processed = False)
        B = self.ds2.get_data('X', processed = False)
        
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return corr_discrepancy(A, B)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A.sample(frac = min_sample_num/ds1_shape[0], replace = True)
                B1 = B.sample(frac = min_sample_num/ds2_shape[0], replace = True)
                return corr_discrepancy(A1, B1)
          
    def dcov(self):
        """ Compute the distance correlation between ds1 and ds2.
        """
        #self.process_data()
        
        A = self.ds1.get_data('X', processed = False)
        B = self.ds2.get_data('X', processed = False)
        
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return cov_discrepancy(A, B)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A.sample(frac = min_sample_num/ds1_shape[0], replace = True)
                B1 = B.sample(frac = min_sample_num/ds2_shape[0], replace = True)
                return cov_discrepancy(A1, B1)
           
    def ks_test(self):
        #self.process_data()
        A = self.ds1.get_data('X', processed=False).values
        B = self.ds2.get_data('X', processed=False).values
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return ks_test(A, B)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A[np.random.choice(A.shape[0], min_sample_num, replace=True)]
                B1 = B[np.random.choice(B.shape[0], min_sample_num, replace=True)]
                return ks_test(A1, B1)
            
    def nn_discrepancy(self):
        #self.process_data()
        A = self.ds1.get_data('X', processed=False).values
        B = self.ds2.get_data('X', processed=False).values
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return nn_discrepancy(A, B)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A[np.random.choice(A.shape[0], min_sample_num, replace=True)]
                B1 = B[np.random.choice(B.shape[0], min_sample_num, replace=True)]
                return nn_discrepancy(A1, B1)
            
    def relief_divergence(self):
        #self.process_data()
        A = self.ds1.get_data('X', processed=False).values
        B = self.ds2.get_data('X', processed=False).values
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                return relief_divergence(A, B)
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A[np.random.choice(A.shape[0], min_sample_num, replace=True)]
                B1 = B[np.random.choice(B.shape[0], min_sample_num, replace=True)]
                return relief_divregence(A1, B1)
  
    def t_test(self):
        """ Perform Student's t-test.
        """
        return ttest_ind(self.ds1.get_data('X'), self.ds2.get_data('X'))
         
    def compute_descriptors(self, norm='manhattan', processed=False):
        """ 
            Compute distances between descriptors of ds1 and ds2.
            
            :param norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
        """
        self.process_data()
        self.ds1.compute_descriptors(processed=True)
        self.ds2.compute_descriptors(processed=True)
        
        descriptors1 = self.ds1.descriptors
        descriptors2 = self.ds2.descriptors
        
        # For each descriptor
        for k in list(descriptors1.keys()):
            if k in descriptors2.keys():
                # Distance
                self.descriptors_dist[k] = distance(descriptors1[k], descriptors2[k], norm=norm)
            
    def compute_comparison_matrix(self, processed=True):
        """ 
            Compute a pandas DataFrame
            Columns: data features
            Rows: univariate comparison metrics (numerical or categorical)
            
            :param processed: If True, processed data are used.
        """
        self.process_data()
        data1 = self.ds1.get_data('X', processed=True)
        data2 = self.ds2.get_data('X', processed=True)
        
        columns = data1.columns.values
        for i, column in enumerate(columns):
        
            # Numerical
            if self.ds1.feat_type[i] == 'Numerical':
                self.comparison_matrix.at['Kolmogorov-Smirnov', column] = kolmogorov_smirnov(data1[column], data2[column])
            
            # Categorical, other
            else:
                f1 = frequency(data1[column])
                f2 = frequency(data2[column])
                
                self.comparison_matrix.at['Kullback-Leibler divergence', column] = kullback_leibler(f1, f2)
                self.comparison_matrix.at['Mutual information', column] = mutual_information(f1, f2)
                self.comparison_matrix.at['Jensen-Shannon divergence', column] = jensen_shannon(f1, f2)
                #self.comparison_matrix.at['Chi-square', column] = chi_square(f1, f2)
                
    def classify(self, clf=LogisticRegression(), processed=True):
        """ Return the scores (classification report: precision, recall, f1-score) of a classifier train on the data labeled with 0 or 1 according to their original dataset.
            
            :param clf: the classifier. It has to have fit(X,y) and score(X,y) methods.
            :param processed: If True, processed data are used.
            :return: Classification report (precision, recall, f1-score).
            :rtype: str
        """
        
        ds1_train = self.ds1.get_data('X_train', processed=True)
        ds1_test = self.ds1.get_data('X_test', processed=True)
        ds2_train = self.ds2.get_data('X_train', processed=True)
        ds2_test = self.ds2.get_data('X_test', processed=True)
    
        # Train set
        X1_train, X2_train = list(ds1_train.values), list(ds2_train.values)
        X_train = X1_train + X2_train
        y_train = [0] * len(X1_train) + [1] * len(X2_train)
        
        # Shuffle
        combined = list(zip(X_train, y_train))
        random.shuffle(combined)
        X_train[:], y_train[:] = zip(*combined)
        
        # Test set
        X1_test, X2_test = list(ds1_test.values), list(ds2_test.values)
        X_test = X1_test + X2_test
        y_test = [0] * len(X1_test) + [1] * len(X2_test)
        
        # Training
        clf.fit(X_train, y_train)
        
        # Score
        #clf.score(X_test, y_test)
        target_names = ['Dataset 1', 'Dataset 2']
        return classification_report(clf.predict(X_test), y_test, target_names=target_names)
        
    def show_classifier_score(self, clf=LogisticRegression()):
        """ Display the scores (classification report: precision, recall, f1-score) of a classifier train on the data labeled with 0 or 1 according to their original dataset.
            (return of 'classify' method)
            
            :param clf: the classifier. It has to have fit(X,y) and score(X,y) methods.
        """
        report = self.classify(clf=clf) #.round(5)
        
        print(clf)
        print('\n')
        print(report)
        print('\n')
          
    def show_descriptors(self):
        """ Show descriptors distances between ds1 and ds2.
        """
        for k in list(self.descriptors_dist.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.descriptors_dist[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value))

    def show_comparison_matrix(self, processed=True):
        """ Display inter-columns comparison.
        """
        if self.comparison_matrix is None:
            self.comparison_matrix = pd.DataFrame(columns=self.ds1.get_data('X', processed=True).columns.values)
            self.compute_comparison_matrix(processed=True)
            
        display(self.comparison_matrix)


    def compute_mda(self, norm='manhattan', precision=0.2, threshold=0.2, area='simpson'):
        """ Compute the accumulation of minimum distances from one dataset to other.
            Use for privacy/resemblance metrics.
            
            :param norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
            :param precision: Curve sampling rate.
            :param threshold: Privacy/resemblance threshold distance (list to compute several for various threshold).
            :param area: 'simpson', 'trapezoidal'
        """
        # Distributions
        A = self.ds1.get_data('X', processed=True, array=True)
        B = self.ds2.get_data('X', processed=True, array=True)
        
        # Distances to nearest neighbors
        mdA, mdB = minimum_distance(A, B, norm=norm)
        
        # Compute for several theshold without re-computing distances
        if isinstance(threshold, list) or isinstance(threshold, np.ndarray):
            res = []
            for t in threshold:
                res.append(compute_mda(mdA+mdB, precision=precision, threshold=t, area=area))
               
            return res
        
        # Curve and metrics
        # Symmetrize
        self.mda = compute_mda(mdA+mdB, precision=precision, threshold=threshold, area=area)
        return self.mda
        
    
    def show_mda(self, save=None):
        """ Show the accumulation of minimum distances from one dataset to other.
            Use for privacy/resemblance metrics
        """
        if self.mda is None:
            self.compute_mda()
            
        (xA, yA), (privacyA, resemblanceA), thresholdA = self.mda
        #(xB, yB), (privacyB, resemblanceB), thresholdB = self.mda2
        
        # Plot A
        print('Nearest neighbors metric')
        plt.plot(xA, yA)
        plt.axvline(x=thresholdA, color='r', label='threshold')
        plt.xlabel('Distance d')
        plt.ylabel('Number of minimum distance < d')
        plt.title('Symmetric MDA')
        plt.legend()
        
        if save is not None:
            plt.savefig(save)
        plt.show()
        
        printmd('** Privacy: **' + str(privacyA))
        printmd('** Resemblance: **' + str(resemblanceA))
        
        
    def show_mda_threshold(self, save=None):
        """ Show privacy and resemblance scores over various threshold.
        """
        ps = [] # privacy scores
        rs = [] # resemblance scores
        ts = np.arange(0.1, 1, 0.1) # thresholds
        
        scores = self.compute_mda(threshold=ts)
        for score in scores:
            (x, y), (privacy, resemblance), threshold = score
            ps.append(privacy)
            rs.append(resemblance)
        
        # Plot    
        plt.plot(ts, ps, label='privacy')
        plt.plot(ts, rs, label='resemblance')
        plt.ylabel('scores')
        plt.xlabel('threshold')
        plt.legend()
        
        if save is not None:
            plt.savefig(save)
        plt.show()
        
     
    def show_mmd(self, alpha = 0.2):
        """ Compute and show MMD between ds1 and ds2
        """
        #self.process_data()
        
        A = self.ds1.get_data('X', processed = False)
        B = self.ds2.get_data('X', processed = False)
        
        ds1_shape = A.shape
        ds2_shape = B.shape
        
        if ds1_shape[1] != ds2_shape[1]:
            printmd('Datasets have different number of features!')
        else:
            try:
                score = mmd(A, B, alpha)
                print('Maximum mean discrepancy: ' + str(score))
            except ValueError:
                printmd('Number of samples unequal in datasets. Selecting minimum number of equal samples randomly')
                min_sample_num = min(ds1_shape[0], ds2_shape[0])
                A1 = A.sample(frac = min_sample_num/ds1_shape[0], replace = True)
                B1 = B.sample(frac = min_sample_num/ds2_shape[0], replace = True)
                score = mmd(A1, B1, alpha)
                print('Maximum mean discrepancy: ' + str(score))
                
         
    def show_pca(self, processed=False, i=1, j=2, verbose=False, save=None, label1='Dataset 1', label2='Dataset 2', size=1, **kwargs):
        """ Compute and show 2D PCA of both datasets on the same plot.
        
            :processed: Get processed data or not (boolean)
            :param i: i_th component of the PCA
            :param j: j_th component of the PCA
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for PCA (see sklearn doc)
        """
        
        X1 = self.get_ds1().get_data(processed=processed)
        X2 = self.get_ds2().get_data(processed=processed)
        
        # Same number of example in both dataset to plot
        if X1.shape[0] < X2.shape[0]:
            X2 = X2.sample(n=X1.shape[0])
        if X1.shape[0] > X2.shape[0]:
            X1 = X1.sample(n=X1.shape[0])
        
        pca1, X1 = compute_pca(X1, verbose, **kwargs)
        pca2, X2 = compute_pca(X2, verbose, **kwargs)
        
        plt.scatter(X1.T[0], X1.T[1], alpha=.9, lw=2, s=size, color='blue', marker='o', label=label1)
        plt.scatter(X2.T[0], X2.T[1], alpha=.8, lw=2, s=size, color='orange', marker='x', label=label2)
        
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        
        plt.xlabel('PC '+str(i))
        plt.ylabel('PC '+str(j))
        plt.title('Principal Component Analysis: PC{} and PC{}'.format(str(i), str(j)))
        
        if save is not None:
            plt.savefig(save)
        plt.show()
        
        
    def show_lda(self, target, processed=False, verbose=False, save=None, **kwargs):
        """ Compute and show 2D LDA of both datasets on the same plot.
        
            :processed: Get processed data or not (boolean)
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for PCA (see sklearn doc)
        """
        
        X1 = self.get_ds1().get_data(processed=processed)
        X2 = self.get_ds2().get_data(processed=processed)
        
        Y1 = X1[target]
        Y2 = X2[target]
        X1.drop(target, axis=1)
        X2.drop(target, axis=1)
        
        lda1, X1 = compute_lda(X1, Y1, verbose, **kwargs)
        lda2, X2 = compute_lda(X2, Y2, verbose, **kwargs)
        
        plt.scatter(X1.T[0], X1.T[1], alpha=.6, lw=2, s=1, color='blue', label='Dataset 1')
        plt.scatter(X2.T[0], X2.T[1], alpha=.2, lw=2, s=1, color='orange', label='Dataset 2')
        
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        
        plt.xlabel('')
        plt.ylabel('')
        plt.title('LDA')
        
        if save is not None:
            plt.savefig(save)
        plt.show()
        

    def compare_marginals(self, metric='all', processed=False, target=None, save=None):
        """ Plot the metric for each variable from ds1 and ds2
            Mean, standard deviation or correlation with target.
        
            :param metric: 'mean', 'std', 'corr', 'all'
            :param target: column name for the target for correlation metric
        """
        if (metric == 'all' or metric == 'corr') and target is None:
            raise OSError('You have to define a target to use {} metric.')
        
        X1 = self.get_ds1().get_data(processed=processed)
        X2 = self.get_ds2().get_data(processed=processed)
        
        x_mean, y_mean = [], []
        x_std, y_std = [], []
        x_corr, y_corr = [], []
        
        
        if metric in ['mean', 'all']:
            for column in list(X1.columns):
                x_mean.append(X1[column].mean())
                y_mean.append(X2[column].mean())
                
        if metric in ['std', 'all']:
            for column in list(X1.columns):
                x_std.append(X1[column].std())
                y_std.append(X2[column].std())
                
        if metric in ['corr', 'all']:  
            if ('y' in self.get_ds1().subsets) and (target is None):
                y1 = self.get_ds1().get_data(s='y', processed=processed)
                y2 = self.get_ds2().get_data(s='y', processed=processed)
            else:
                y1 = X1[target]
                y2 = X2[target]
                
            # Flatten one-hot (dirty)
            if len(y1.shape) > 1:
                if y1.shape[1] > 1:
                    y1 = np.where(y1==1)[1]
                    y1 = pd.Series(y1)
            if len(y2.shape) > 1:
                if y2.shape[1] > 1:
                    y2 = np.where(y2==1)[1]
                    y2 = pd.Series(y2)
                
            for column in list(X1.columns):  
                x_corr.append(X1[column].corr(y1))
                y_corr.append(X2[column].corr(y2))
                
        elif metric not in ['mean', 'std', 'corr', 'all']:
            raise OSError('{} metric is not taken in charge'.format(metric))
                
        if metric == 'mean':
            plt.plot(x_mean, y_mean, 'o', color='b')
            plt.xlabel('Mean of variables in dataset 1')
            plt.ylabel('Mean of variables in dataset 2')
            
        elif metric == 'std':
            plt.plot(x_std, y_std, 'o', color='g')
            plt.xlabel('Standard deviation of variables in dataset 1')
            plt.ylabel('Standard deviation of variables in dataset 2')
            
        elif metric == 'corr':
            plt.plot(x_corr, y_corr, 'o', color='r')
            plt.xlabel('Correlation with target of variables in dataset 1')
            plt.ylabel('Correlation with target of variables in dataset 2')            
        
        elif metric == 'all':
            plt.plot(x_mean, y_mean, 'o', color='b', alpha=0.9, label='Mean')
            plt.plot(x_std, y_std, 'o', color='g', alpha=0.8, label='Standard deviation')
            plt.plot(x_corr, y_corr, 'o', color='r', alpha=0.7, label='Correlation with target')
            plt.xlabel('Dataset 1 variables')
            plt.ylabel('Dataset 2 variables')
            plt.legend(loc='upper left')
            plt.ylim(-1, 1)
            plt.xlim(-1, 1)            
            
        else:
            raise OSError('{} metric is not taken in charge'.format(metric))
        
        plt.plot([-1, 1], [-1, 1], color='grey', alpha=0.4)
        
        if save is not None:
            plt.savefig(save)
        plt.show()
