import logging
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


@step
def clean_df(df: pd.DataFrame) -> None:
    pass