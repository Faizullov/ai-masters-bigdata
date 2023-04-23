#!/opt/conda/envs/dsenv/bin/python

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import TransformerMixin #gives fit_transform method for free
from sklearn.metrics import log_loss

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

#
# Import model definition
#

import mlflow

#
# Logging initialization

logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#

class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 

#
# Dataset fields
#
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

fields = ["id", "label"] + numeric_features + categorical_features
#
# Model pipeline
#


# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('drop', columnDropperTransformer(categorical_features))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', 'drop', categorical_features)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LogisticRegression(max_iter=int(sys.argv[2])))
])

try:
  train_path = sys.argv[1]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_PATH {train_path}")

#
# Read dataset
#
#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['label']), df['label'], test_size=0.9, random_state=42
)

#
# Train the model
#
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="I_LOVE_BIGDATA"):
    # with mlflow.start_run(run_name="I LOVE BIGDATA"):
    model.fit(X_train, y_train)

    model_score = log_loss(y_test, model.predict(X_test))

    # logging.info(f"model score: {model_score:.3f}")

    mlflow.log_param("model_param1", sys.argv[2])
    mlflow.sklearn.log_model(model, artifact_path="models")
    mlflow.log_metric("log_loss", model_score)

