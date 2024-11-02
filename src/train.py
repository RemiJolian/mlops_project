import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from mlflow.models import infer_signature
import os



#mlflow.set_tracking_uri("My-Machine-Learning-Projects\MLFlow\mlruns")

# Set MLflow tracking URI
mlflow.set_tracking_uri("CI-CD-for-ML-Models\mlops_project\src\mlruns")

# Optionally set experiment name, which will create a new experiment if it doesn't exist
mlflow.set_experiment("Iris_Classification2")

# Load sample data (Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow tracking
with mlflow.start_run(run_name="Unique_Experiment_Run"):
    # Set parameters for the model
    n_estimators = 100
    max_depth = 5

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train, y_train)

    # Get model predictions
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Create model signature
    signature = infer_signature(X_train, predictions)

    # Log model with signature and input example
    mlflow.sklearn.log_model(
        rf, 
        "random_forest_model",
        signature=signature,
        input_example=X_train[:5]  # Using first 5 rows as example
    )

    # Log accuracy metric
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")

    # Print run ID and experiment ID to confirm tracking
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.active_run().info.experiment_id
    print(f"Run ID: {run_id}")
    print(f"Experiment ID: {experiment_id}")

    # Use an absolute path to a common directory
mlflow.set_tracking_uri("file:///absolute/path/to/shared/mlruns")


# ---------------------------------------------------------------------------------------------------------
# Code 2:

# import mlflow
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_iris
# from mlflow.models import infer_signature

# # Set tracking URI to shared location
# mlflow.set_tracking_uri("file:/mlops_project/mlruns")
# mlflow.set_experiment("Iris_Classification")

# def train():
#     data = load_iris()
#     X, y = data.data, data.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     with mlflow.start_run():
#         model = RandomForestClassifier(n_estimators=100, max_depth=5)
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, predictions)
#         signature = infer_signature(X_train, predictions)

#         mlflow.log_param("n_estimators", 100)
#         mlflow.log_param("max_depth", 5)
#         mlflow.log_metric("accuracy", accuracy)

#         mlflow.sklearn.log_model(
#             model,
#             "model",
#             signature=signature,
#             input_example=X_train[:5]
#         )
#         print(f"Model logged with accuracy: {accuracy}")

# if __name__ == "__main__":
#     train()