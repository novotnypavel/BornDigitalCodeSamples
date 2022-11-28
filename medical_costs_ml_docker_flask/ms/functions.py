import pandas as pd
from ms import model
import pickle
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder 

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(X):
    #X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, model)
    return {
        'status': 200,
        'prediction': int(prediction)
    }

def preprocess_data(json_data):
    data = pd.DataFrame.from_dict(json_data)

    which_categ = data.dtypes == 'object' 
    categ_names = list(data.columns[which_categ])

    age_binned = pd.cut(
        x=data['age'],
        bins=[0,30,45,200],
        labels=["young", "middle", "not young"]
    )

    age_disc_dict = {
        "young":1,
        "middle":2, 
        "not young":3
    }
    age_binned_le = age_binned.map(age_disc_dict)

    data = data.drop(['age'],axis=1)
    data['age_group'] = age_binned_le
    

    with open(r"model/insurance_train_test.pickle", "rb") as input_file:
        X_train, y_train, X_test, y_test = pickle.load(input_file) 

    for c in categ_names: 
        encoder = TargetEncoder() 
        encoder.fit(data[c], y_train)
        data[c+'_mean'] = encoder.transform(data[c]) 

    data = data.drop(categ_names, axis=1)
    stdsc = StandardScaler()
    stdsc.fit(X_train)
    data = stdsc.transform(data)
    return data

