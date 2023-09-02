from zenml import pipeline
from customerSatisfaction.steps.ingest_data import ingest_df
from customerSatisfaction.steps.clean_data import clean_df
from customerSatisfaction.steps.model_train import train_model
from customerSatisfaction.steps.evaluation import evaluate_model


@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)
    