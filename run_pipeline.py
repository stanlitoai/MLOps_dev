from src.customerSatisfaction.pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = "data/olist_customers_dataset.csv")
    # mlflow ui --backend-store-uri "file:/home/gitpod/.config/zenml/local_stores/33732d56-f8e4-4644-a07a-15cf6c648ddb/mlruns"