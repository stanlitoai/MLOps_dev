import logging
import pandas as pd
from zenml import step
from customerSatisfaction.config.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing import Tuple
from typing_extensions import Annotated
logger = logging.getLogger(__name__)


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Cleans the data and splits it into training and test
    Args:
        df (pd.DataFrame): Dataframe to be cleaned
    Returns:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        return X_train, X_test, y_train, y_test
        logger.info("Data cleaning completed")
        
    except Exception as e:
        logger.error("Error cleaning data : ".format(e))
        raise e
        
        
        