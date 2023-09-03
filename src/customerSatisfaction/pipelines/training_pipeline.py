from zenml import pipeline
from customerSatisfaction.steps.ingest_data import ingest_df
from customerSatisfaction.steps.clean_data import clean_df
from customerSatisfaction.steps.model_train import train_model
from customerSatisfaction.steps.evaluation import evaluate_model
import logging


@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    logging.info("Done ingest_df pipeline")
    X_train, X_test, y_train, y_test = clean_df(df)
    logging.info("Done clean_df pipeline")
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2_score, rmse = evaluate_model(model, X_test, y_test)
    