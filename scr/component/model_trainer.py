import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (RandomForestRegressor, 
                              AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from scr.exception import CutomException
from scr.logger import logging
from scr.utils import save_object, evaluate_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and testing data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbor Regression": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "GradientBoosting Regression": GradientBoostingRegressor(),
                "XGBRFRegressor": XGBRFRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CutomException("No Best Model Found")
            logging.info("Best Model found in both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CutomException(e, sys)
