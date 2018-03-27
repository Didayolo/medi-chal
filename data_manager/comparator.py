# Imports
from scipy.stats import ttest_ind

class Comparator():
    def __init__(self, ds1, ds2):
        """
            Constructor
            ds1 and ds2 are AutoML objects
            
        """
        # Datasets to compare
        self.ds1 = ds1
        self.ds2 = ds2
        
        # Dictionary of distances between each descriptor of ds1 and ds2
        self.descriptors_dist = dict()
        self.compare_descriptors()

        assert(ds1.info['feat_num'] == ds2.info['feat_num'])

    def t_test(self):
        """
            Perform Student's t-test.
        """
        return ttest_ind(self.ds1.data['X'], self.ds2.data['X'])
         
    def compare_descriptors(self):
        """ Idea : automatic comparison between descriptors
        """
        descriptors1 = self.ds1.get_descriptors()
        descriptors2 = self.ds2.get_descriptors()
        # For each descriptor
        for k in list(descriptors1.keys()):
            # Distance
            self.descriptors_dist[k] = abs(descriptors1[k] - descriptors2[k])
            
    def show_descriptors(self):
        """ Show descriptors distances between ds1 and ds2
        """
        for k in list(self.descriptors_dist.keys()):
            key = k.capitalize().replace('_', ' ')
            value = self.descriptors_dist[k]
            if isinstance(value, str):
                value = value.capitalize().replace('_', ' ').replace('.', ' ')

            print('{}: {}'.format(key, value))
            
