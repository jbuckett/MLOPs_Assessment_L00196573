import pandas as pd
import numpy as np
import mlflow
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

# If running in GitHub Actions, use a local folder backup otherwise use VM server
if os.environ.get("GITHUB_ACTIONS") == "true":
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5555")

# load the life expectancy dataset
df = pd.read_csv("Life Expectancy Data.csv")

# keep only the columns I want to use
df = df[['Status', 'Schooling', 'Life expectancy ']]

# drop any rows that have missing data to keep things simple
df = df.dropna()

# change Status from text to numbers (Developing = 0, Developed = 1)
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})

# set features (X) and predict (y) life expectancy
X = df[["Status", "Schooling"]]
y = df["Life expectancy "]

# split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# start recording this run in MLflow
with mlflow.start_run():
    # train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions to test it
    predictions = model.predict(X_test)

 # calculate how good the model is
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # save these metrics to MLflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # log the model to MLflow
    result = mlflow.sklearn.log_model(model, "model")

    # register the model
    mlflow.register_model(
        model_uri=result.model_uri,
        name="life-expectancy-model"
    )

    # save the model to a file the  flask app can use it
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

print("Model trained and saved as model.pkl!")
