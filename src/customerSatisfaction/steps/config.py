from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """
    Parameters for the `sklearn.linear_model.LogisticRegression` step.
    """
    model_name: str = 'LinearRegression'
    # model_class = 'LogisticRegression'
    # hyperparameter_ranges = {
    #     'penalty': ['l1', 'l2'],
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #    'solver': ['liblinear', 'newton-cg', 'lbfgs','sag','saga'],
    #    'max_iter': [100, 100, 1000]}
    
    