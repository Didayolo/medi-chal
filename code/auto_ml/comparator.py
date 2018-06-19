# Imports
from utilities import *
from scipy.stats import ttest_ind
from IPython.display import display
from metric import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random
from encoding import frequency
from auto_ml import AutoML

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
        self.mda1 = None
        self.mda2 = None

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
        data1 = self.ds1.get_data('X', processed=True).values
        data2 = self.ds2.get_data('X', processed=True).values
        return distance(data1, data2, axis=axis, norm=norm)

    def dcov(self):
        """ Compute the distance correlation between ds1 and ds2.
        """
        return distcorr(self.ds1.get_data('X'), self.ds2.get_data('X'))

    def t_test(self):
        """ Perform Student's t-test.
        """
        return ttest_ind(self.ds1.get_data('X'), self.ds2.get_data('X'))
         
    def compute_descriptors(self, norm='manhattan', processed=False):
        """ 
            Compute distances between descriptors of ds1 and ds2.
            
            :param norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
        """
        
        self.ds1.compute_descriptors(processed=processed)
        self.ds2.compute_descriptors(processed=processed)
        
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
        
        data1 = self.ds1.get_data('X', processed=processed)
        data2 = self.ds2.get_data('X', processed=processed)
        
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
        
        ds1_train = self.ds1.get_data('X_train', processed=processed)
        ds1_test = self.ds1.get_data('X_test', processed=processed)
        ds2_train = self.ds2.get_data('X_train', processed=processed)
        ds2_test = self.ds2.get_data('X_test', processed=processed)
    
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
            self.comparison_matrix = pd.DataFrame(columns=self.ds1.get_data('X', processed=processed).columns.values)
            self.compute_comparison_matrix(processed=processed)
            
        display(self.comparison_matrix)


    def compute_mda(self, norm='manhattan', precision=0.2, threshold=None, area='simpson'):
        """ Compute the accumulation of minimum distances from one dataset to other.
            Use for privacy/resemblance metrics.
            
            :param norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
            :param precision: Curve sampling rate.
            :param threshold: Privacy/resemblance threshold distance.
            :param area: 'simpson', 'trapezoidal'
        """
        # Distributions
        A = self.ds1.get_data('X', processed=True, array=True)
        B = self.ds2.get_data('X', processed=True, array=True)
        
        # Distances to nearest neighbors
        mdA, mdB = minimum_distance(A, B, norm=norm)
        
        # Curve and metrics
        self.mda1 = compute_mda(mdA, precision=precision, threshold=threshold, area=area)
        self.mda2 = compute_mda(mdB, precision=precision, threshold=threshold, area=area)
        
    
    def show_mda(self):
        """ Show the accumulation of minimum distances from one dataset to other.
            Use for privacy/resemblance metrics
        """
        if self.mda1 is None:
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
        plt.legend()
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
        plt.legend()
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
