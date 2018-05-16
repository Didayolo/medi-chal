# Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
problem_dir = 'code/auto_ml'  
from sys import path
path.append(problem_dir)
from auto_ml import AutoML

class RF_generator():
    def __init__(self, ds):
        """ Data generator using multiple imputations with random forest
            Input:
              ds: AutoML object containing data
        """
        # List of Random Forests
        self.models = []
        
        # Random forest from sklearn
        self.regressor = RandomForestRegressor
        self.classifier = RandomForestClassifier
        
        # AutoML dataset
        self.ds = ds
    
    
    def get_data(self):
        return self.ds.get_processed_data()['X']
    
    
    def fit(self, **kwargs):
        """ Fit one random forest for each column, given the others
            Input:
              kwargs: Random Forest parameters
        """
        data = self.get_data()

        for i in range(len(data.columns)):
            # May bug with duplicate names in columns
            y = data[data.columns[i]]
            X = data.drop(data.columns[i], axis=1) 
            
            # Regressor or classifier
            if self.ds.is_numerical[i] == 'numerical':
                model = self.regressor(**kwargs)
            else:
                model = self.classifier(**kwargs)
            
            model.fit(X, y)
            self.models.append(model)
      
        
    def generate(self, p=0.5):
        """ Generate examples by copying data and then do values imputations
            Input:
              p: the probability of changing a value
                 if p=0, the generated dataset will be equals to the original
                 if p=1, the generated dataset will contains only new values
            Return:
              new_data: Generated pandas DataFrame
        """
        data = self.get_data()
        
        new_data = data.copy()
        
        for x in list(data.index.values):
            for i, y in enumerate(list(data.columns.values)):
            
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    new_data.at[x, y] = self.models[i].predict(row)

        return new_data
    
    
    def generate_to_automl(self, input_dir, basename, p=0.5):
        """ Generate a DataFrame and save it in automl format
            Input:
              input_dir
              basename
              p
            Return: AutoML object
        """
        X = self.generate(p=p)
        return AutoML.from_df(input_dir, basename, X, y=None)
