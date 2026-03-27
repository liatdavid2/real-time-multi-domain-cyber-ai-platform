
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    mlflow.set_experiment("network_training")

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print(classification_report(y_test, preds))

        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
