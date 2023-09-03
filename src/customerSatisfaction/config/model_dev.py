import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """Abstract class for models"""
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): training data
            y_train (pd.Series): training labels
        
        """
        pass
    
class LinearRegressionModel(Model):
    """Linear regression model"""
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): training data
            y_train (pd.Series): training labels
        
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed successfully")
            return reg
        except Exception as e:
            logging.error(e)
            logging.error("Model training failed")
            raise e
        

    