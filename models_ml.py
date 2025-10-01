from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

@dataclass
class SKModel:
    name: str
    model: object

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return SKModel(name="LinearRegression", model=lr)

def train_random_forest(X_train, y_train, n_estimators=300, max_depth=None, n_jobs=-1):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return SKModel(name="RandomForest", model=rf)

def predict_model(skmodel: SKModel, X_test):
    return skmodel.model.predict(X_test)
