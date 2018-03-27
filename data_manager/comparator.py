
from scipy.stats import ttest_ind

class Comparator():
    def __init__(self, ds1, ds2):
        """
            Constructor
            
        """
        self.ds1 = ds1
        self.ds2 = ds2

        assert(ds1.info['feat_num'] == ds2.info['feat_num'])

    def t_test(self):
        return ttest_ind(self.ds1.data['X'], self.ds2.data['X'])

