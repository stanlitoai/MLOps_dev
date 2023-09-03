import logging

import mlflow
import pandas as pd
from customerSatisfaction.config.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from zenml import step
from .config import ModelNameConfig
# from zenml.client import Client



logger = logging.getLogger(__name__)

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
    ) -> RegressorMixin:
    
    """Trains a model on the data.
    Args:
    X_train (pd.DataFrame): Training data.
    X_test (pd.DataFrame): Test data.
    y_train (pd.DataFrame): Training labels.
    y_test (pd.DataFrame): Test labels.
    Returns:
    RegressorMixin: Trained model.
    
    """
    try:
        model = None
        tuner = None
        if config.model_name = "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        elif config.model_name = "RandomForest":
            model = RandomForestModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        elif config.model_name = "XGBoost":
            model = XGBoostModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        elif config.model_name = "LightGBM":
            model = LightGBMModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        elif config.model_name = "HyperparameterTuner":
            tuner = HyperparameterTuner()
            trained_model = tuner.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"{config.model_name} is not a valid model")
        
    except Exception as e:
        logger.error("Error while training model: {}".format(e))
        raise e
            