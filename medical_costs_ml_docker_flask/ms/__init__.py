from flask import Flask
#import joblib
import pickle

filename = 'model/rf_model.sav'
# Initialize App
app = Flask(__name__)

# Load models
#model = joblib.load('model/model_binary.dat.gz')
model = pickle.load(open(filename, 'rb'))

