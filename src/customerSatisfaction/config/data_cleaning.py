import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """Abstract class for training"""
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
    
class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessingb data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            logging.info("Dataprocessing completed successfully")
            return data
               
        except Exception as e:
            logging.error("Error in processing data: {}".format(e))
            raise e
        



class DataSplitStrategy(DataStrategy):
    """
    Data spliting strategy which split the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Data splitting strategy which splits the data into train and test data.
        """
        try:
            logging.info("Data splitting strategy")
            
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in Splitting  data: {}".format(e))
            


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and split it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        
        """
        try:
            logging.info("handle_data called with data type ")
            
            return self.strategy.handle_data(self.data)
        except Exception as e:   
            logging.error("Error in handling data: {}".format(e))
            raise e