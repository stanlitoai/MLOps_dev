import logging
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)

@step
def train_model(df: pd.DataFrame) -> None:
    """Trains a model on the data.
    """
    pass