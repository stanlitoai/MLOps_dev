import logging
import os
import pandas as pd
from zenml import step

class IngestData:
    """Ingest data from
    Args:
        data_path (str): path to data file
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    
    def ingest_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
        
@step(name="ingest_data")       
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingest data from
    Args:
        data_path (str): path to data file
    Returns:
        pd.DataFrame: dataframe of data
    """
    try:
        # logging.info(f"Ingesting data from {data_path}")
        ingest_data_step = IngestData(data_path)
        df = ingest_data_step.ingest_data()
        return df
    except Exception as e:
        logging.error(f"Error ingesting data from {data_path}: {e}")
        raise e