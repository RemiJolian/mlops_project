# src/train.py
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from mlflow.models import infer_signature

# Set tracking URI to shared location
mlflow.set_tracking_uri("file:/mlops_project/mlruns")
mlflow.set_experiment("Iris_Classification")

def train():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        signature = infer_signature(X_train, predictions)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5]
        )
        print(f"Model logged with accuracy: {accuracy}")

if __name__ == "__main__":
    train()
