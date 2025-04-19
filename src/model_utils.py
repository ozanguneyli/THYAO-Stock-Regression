import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
        "KNeighbors": KNeighborsRegressor(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "MLPRegressor": MLPRegressor(random_state=42, max_iter=1000),
        "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
        "LGBMRegressor": LGBMRegressor(random_state=42)
    }

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    models = get_models()
    results = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        cv_rmse = np.sqrt(-cv_scores).mean()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2
        })
    return pd.DataFrame(results), models, X_train, X_test, y_train, y_test