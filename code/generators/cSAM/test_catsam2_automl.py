#!/usr/bin/python

from catSAM2 import catSAM
import pandas as pd

# Import AutoML
problem_dir = '../auto_ml'  
from sys import path
path.append(problem_dir)
from auto_ml import AutoML

# Load AutoML files
data_path = '../../data/'
input_dir = 'adult'
basename = 'adult'
D = AutoML(data_path+input_dir, basename)

# Identify categorical variables
categorical_variables = [x == 'Categorical' for x in D.feat_type]

# Label encoding (strings not supported)
D.process_data(norm='none', code='label', missing='none')
data = D.get_data(processed=True)

# result = run_SAM(a, batch_size=1000, train_epochs=200, test_epochs=200,
#                  dnh=20, nh=5, lr=0.001, plot_generated_pair=False)

# Train SAM
model = catSAM(lr=0.01, dlr=0.01, nh=10, dnh=50, l1=.005,
               train_epochs=50, test_epochs=50)

# Generate
result, sam, data = model.predict(data, categorical_variables, 
                                  nruns=1, gpus=0, njobs=1, plot=False, return_model=True)

# Generate Data and save it.
df = pd.DataFrame(sam(data).data.numpy()) # columns name...
AutoML.from_df(data_path+input_dir+'_gen_sam', basename+'_gen_sam', df, y=None)
