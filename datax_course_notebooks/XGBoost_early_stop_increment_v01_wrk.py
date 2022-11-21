#isnpired by https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/

import xgboost as xgb #pip3 install xgboost
import pickle
from matplotlib import pyplot

#%reset -f

SEED = 500

#load data (already split into training and validation)
with open(r"insurance_train_test.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

###############################
###Learn full 250 estimators###
###############################

# Instantiate a XGBoost regressor 
params = {'colsample_bytree': 1,
          'learning_rate': 0.1, 
          'max_depth': 3, 
          'n_estimators': 250, #250
          'subsample': 0.8}

#** to provide keyword parameters
#instantiate xgb.XGBRegressor
gbm = xgb.XGBRegressor(**params, objective='reg:squarederror', seed = SEED)

#We provide an array of X and y pairs to the eval_metric argument when fitting our XGBoost model. 
#In addition to a test set, we can also provide the training dataset. This will provide a report on how well the model is performing on both training and test sets during training.
eval_set = [(X_train, y_train), (X_test, y_test)]
gbm.fit(X_train, y_train, eval_set=eval_set, verbose=True)

#Plot train and test curve
#evaluation stored in gbm.evals_result()
#retrieve performance metrics
results = gbm.evals_result() #evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

#Plot rmse vs nr estimators
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('rmse')
pyplot.title('XGBoost RMSE')
pyplot.show()

####################
###Early stopping###
####################

#Check for no improvement in rmse over the 10 epochs
#If the hold-out metric ("rmse" in our case) does not improve for a given number of rounds (early_stopping_rounds), training is terminanted
#set 10 early_stopping_rounds
gbm.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=True)
print(gbm.best_iteration) #best_iteration with lowest RMSE

##########################
###Incremental learning###
##########################

from sklearn.model_selection import train_test_split
#split training data into 2 halves
X_train_batch1, X_train_batch2, y_train_batch1, y_train_batch2= train_test_split(
    X_train, 
    y_train, 
    test_size=0.5, #0.5
    random_state=SEED
    )

#instantiate (same as above)
gbm_increment = xgb.XGBRegressor(**params, objective='reg:squarederror',seed = SEED)
gbm_increment.fit(X_train_batch1, y_train_batch1, eval_set=eval_set, verbose=True)
#Save the model using save_model function
gbm_increment.save_model('gbm_increment.model')

#X_train_batch1, y_train_batch1 not necessary anymore
del(X_train_batch1, y_train_batch1)

#update model with new parameters use xgb_model = 'gbm_increment.model' to set which model to update
gbm_increment.fit(X_train_batch2, y_train_batch2,xgb_model = 'gbm_increment.model', eval_set=eval_set, verbose=True)
