import logging
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """evaluates the model on the data.
    """
    pass