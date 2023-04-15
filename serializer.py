import pickle, os, joblib

import matplotlib
import pandas
import numpy
import tensorflow
import seaborn
import sklearn
import flask


from simple_heuristic_algorithm import CoverTypeClassifierHeuristic
from two_ml_models import CoverTypeClassifierRFLR
from ann_model import CoverTypeClassifierNN

if not os.path.exists('serialized_files'):
    os.makedirs('serialized_files')

with open('serialized_files/model_heuristic.joblib', 'wb') as f:
    joblib.dump(CoverTypeClassifierHeuristic('dataset_and_info/covtype.data'), f)
    print("heuristic serialized")

with open('serialized_files/model_RFLR.joblib', 'wb') as f:
    joblib.dump(CoverTypeClassifierRFLR('dataset_and_info/covtype.data'), f)
    print("RFLR serialized")

with open('serialized_files/model_NN.joblib', 'wb') as f:
    joblib.dump(CoverTypeClassifierNN('dataset_and_info/covtype.data'), f)
    print("NN serialized")


print("pandas - ", pandas.__version__)
print("numpy - ", numpy.__version__)
print("matplotlib - ", matplotlib.__version__)
print("tensorflow - ", tensorflow.__version__)
print("seaborn - ", seaborn.__version__)
print("scikit-learn - ", sklearn.__version__)
print("flask - ", flask.__version__)

