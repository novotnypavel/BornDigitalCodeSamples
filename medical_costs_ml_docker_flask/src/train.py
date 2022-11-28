#%%
import pandas as pd
import pickle 
import matplotlib.pyplot as plt 
import time 

# Usefull functions
#from sklearn.preprocessing import KBinsDiscretizer #for binning variables
from sklearn.model_selection import train_test_split #for split into training and testing part
from sklearn.model_selection import GridSearchCV

from category_encoders import TargetEncoder #for target (mean) encoding
from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


"""
data
source: https://github.com/stedy/Machine-Learning-with-R-datasets
Individual medical costs billed by health insurance
"""

#%%
# Data load and initial inspection
data_insurance_orig = pd.read_csv('../data/insurance.csv') 

data_insurance_orig.head()
data_insurance_orig.describe(include='all')
data_insurance_orig.dtypes

#find categorical variables
which_categ = data_insurance_orig.dtypes == 'object' #shows data types of columns of the data frame
categ_names = list(data_insurance_orig.columns[which_categ])

# Categorical data
#typecast categorical features to a category dtype for faster performance
for i_var in categ_names:
    data_insurance_orig[i_var] = data_insurance_orig[i_var].astype('category')

print(data_insurance_orig.dtypes == 'category')

#%%
#Copy data_insurance_orig to another variable data_insurance (.copy so really a duplicate is created) 
data_insurance = data_insurance_orig.copy() 
#(you will need the _orig file later)

#%%
###one hot encoding

# I wanted to use OHE but I'd need to resolve ensuring the same feature columns for test data (i.e. when passing individual observation one hot encoding implicitly wouldn't work)
#data_insurance = pd.get_dummies(data_insurance, columns = ["region"])
#data_insurance = pd.get_dummies(data_insurance, columns = ["smoker","sex"], drop_first = False) 
#drop_first argument to set the first category as reference category (dummy vars)

###Ordinal encoding

#Discretize age using pd.cut and apply ordinal encoding to the age classes
age_binned=pd.cut(
    x=data_insurance['age'],
    bins=[0,30,45,200],
    labels=["young", "middle", "not young"]
    )

print(age_binned.value_counts())

#ordinal encoding of age variable
age_disc_dict = {
    "young":1,
    "middle":2, 
    "not young":3
    }
age_binned_le = age_binned.map(age_disc_dict)

data_insurance = data_insurance.drop(['age'],axis=1)
data_insurance['age_group'] = age_binned_le




#%%
# target (mean) encoding
#Apply target encoding to all categorical variables in original dataset
#Split the dataset to training and test set must be performed before encoding
#Watch out for data leak! 


SEED = 500
X_train, X_test, y_train, y_test= train_test_split(
    data_insurance.drop(['charges'], axis = 1), #explanatory
    data_insurance[['charges']], #response
    test_size=0.2, #hold out size
    random_state=SEED
    )

for c in categ_names: 
        encoder = TargetEncoder() 
        encoder.fit(X_train[c], y_train)
        X_train[c+'_mean'] = encoder.transform(X_train[c]) 
        X_test[c+'_mean'] = encoder.transform(X_test[c])

#drop original categorical features since it's now encoded
X_train = X_train.drop(categ_names, axis=1)
X_test = X_test.drop(categ_names, axis=1) 

# Scaling
#%%
stdsc = StandardScaler()

# fit scaler on training data
stdsc.fit(X_train)

#scale both training and testing data
X_train_sc = stdsc.transform(X_train)
X_test_sc = stdsc.transform(X_test)


#%%
# Modelling: RF regressor
forest_model_0 = RandomForestRegressor(n_estimators=5000,
                                   min_samples_split=0.05,
                                   max_features = 4,
                                   random_state=SEED)


#%%
forest_model_0.fit(X_train_sc, y_train.values.ravel()) #values.ravel() flattened array expected by RandomForestRegressor
# Predict the test set labels 'y_pred'
y_pred_0 = forest_model_0.predict(X_test_sc)
# Evaluate the test set RMSE
rmse_test_0 = mean_squared_error(y_test, y_pred_0, squared=False)
print(rmse_test_0)


#%%
# save the model to disk
filename = 'rf_model.sav'
pickle.dump(forest_model_0, open(filename, 'wb'))
 
# some time later...
#%%
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, y_test)


#%% 
# Save testing and training dataset to pickle file

#save testing and training data for later use (use list as a container) to a pickle file
with open(r"../model/insurance_train_test.pickle", "wb") as output_file: #w - write, b 
    pickle.dump([X_train, y_train, X_test, y_test], output_file) #dump - to save the arguments 

#This file will be used in later scripts.
#You can load the file containing variables [X_train, y_train, X_test, y_test]
#%%
#with open(r"../model/insurance_train_test.pickle", "rb") as input_file:
#    X_train_r, y_train_r, X_test_r, y_test_r = pickle.load(input_file) 

