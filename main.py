import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class for data handling strategies"""
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """Strategy for preprocessing data"""
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # ... (unchanged)

class FeatureEngineeringStrategy(DataStrategy):
    """Strategy for feature engineering"""
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # ... (unchanged)

class DataSplitStrategy(DataStrategy):
    """Data splitting strategy which splits the data into train and test data."""
    
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logging.info("Data splitting strategy")
            
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in splitting data: {}".format(e))
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data, performs feature engineering, 
    and splits it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategies: List[DataStrategy]):
        """Initializes the DataCleaning class with a list of strategies."""
        self.data = data
        self.strategies = strategies
        
    def handle_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logging.info("Data cleaning process started")
            
            # Apply data handling strategies sequentially
            for strategy in self.strategies:
                if isinstance(strategy, DataSplitStrategy):
                    # Capture the train and test data when using DataSplitStrategy
                    X_train, X_test, y_train, y_test = strategy.handle_data(self.data)
                else:
                    self.data = strategy.handle_data(self.data)
            
            logging.info("Data cleaning completed successfully")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:   
            logging.error("Error in data cleaning process: {}".format(e))
            raise e

# Example usage:
if __name__ == "__main__":
    # Load your dataset here
    data = pd.read_csv("your_dataset.csv")
    
    # Define the list of data handling strategies
    strategies = [
        DataPreProcessStrategy(),
        FeatureEngineeringStrategy(),
        DataSplitStrategy()
    ]
    
    # Create a DataCleaning instance and apply strategies
    data_cleaner = DataCleaning(data, strategies)
    X_train, X_test, y_train, y_test = data_cleaner.handle_data()
    
    
    # Perform EDA (Exploratory Data Analysis)
    # Example: Plot histograms of numeric features
    numeric_features = X_train.select_dtypes(include=[np.number])
    for col in numeric_features.columns:
        sns.histplot(X_train[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.show()
        



########################################





import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract class for data handling strategies"""
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """Strategy for preprocessing data"""
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Drop unnecessary columns
            columns_to_drop = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
            data = data.drop(columns_to_drop, axis=1)
            
            # Fill missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
            data["review_comment_message"].fillna("No review", inplace=True)
            
            logging.info("Data preprocessing completed successfully")
            
            return data
        except Exception as e:
            logging.error("Error in processing data: {}".format(e))
            raise e

class FeatureEngineeringStrategy(DataStrategy):
    """Strategy for feature engineering"""
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic feature engineering
        """
        try:
            # Add a feature for the total product dimensions
            data["total_product_dimensions"] = (
                data["product_length_cm"]
                * data["product_height_cm"]
                * data["product_width_cm"]
            )
            
            # You can add more feature engineering steps here
            
            logging.info("Feature engineering completed successfully")
            
            return data
        except Exception as e:
            logging.error("Error in feature engineering: {}".format(e))
            raise e

class DataSplitStrategy(DataStrategy):
    """Data splitting strategy which splits the data into train and test data."""
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Split the data into train and test sets
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
            logging.error("Error in splitting data: {}".format(e))
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data, performs feature engineering, 
    and splits it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategies: List[DataStrategy]):
        """Initializes the DataCleaning class with a list of strategies."""
        self.data = data
        self.strategies = strategies
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            logging.info("Data cleaning process started")
            
            # Apply data handling strategies sequentially
            for strategy in self.strategies:
                self.data = strategy.handle_data(self.data)
            
            logging.info("Data cleaning completed successfully")
            
            return self.data
        except Exception as e:   
            logging.error("Error in data cleaning process: {}".format(e))
            raise e

# Example usage:
if __name__ == "__main__":
    # Load your dataset here
    data = pd.read_csv("your_dataset.csv")
    
    # Define the list of data handling strategies
    strategies = [
        DataPreProcessStrategy(),
        FeatureEngineeringStrategy(),
        DataSplitStrategy()
    ]
    
    # Create a DataCleaning instance and apply strategies
    data_cleaner = DataCleaning(data, strategies)
    cleaned_data = data_cleaner.handle_data()
    
    # Perform EDA (Exploratory Data Analysis)
    # Example: Plot histograms of numeric features
    numeric_features = cleaned_data.select_dtypes(include=[np.number])
    for col in numeric_features.columns:
        sns.histplot(cleaned_data[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.show()











