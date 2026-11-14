"""Copy this file to add a model. Every model module exposes NAME, build(), run(data), report(res)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from constants import SEED
from models import common

NAME = "Model name"
GRID = {"clf__n_estimators": [200, 500]}  
# keys must be prefixed with "clf__" (double _) take a look at scikitlearn's GridSearchCV documentation 
#for more details on how to specify hyperparameter grids. You can add more hyperparameters to the grid as needed.


def build():
    return Pipeline([("pre", common.preprocessor()), ("clf", RandomForestClassifier(random_state=SEED))])
# sklearn's Pipeline allows you to chain together multiple steps, such as preprocessing and model training, into a single object. 
# In this case, the pipeline consists of two steps: "pre" for preprocessing the data using the preprocessor() function,
# and "clf" for training a RandomForestClassifier with a specified random seed.

def run(data):  # training data only - the test set is evaluated once in master.ipynb's Final Evaluation section
    return common.fit(NAME, build(), GRID, data)


def report(res):
    common.report(res)  # CV view; add model-specific plots (e.g. feature importances) below
