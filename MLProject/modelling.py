import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# AUTLOG
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv("../Titanic-Dataset_preprocessing.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Prediction
preds = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, preds)

# Log metric tambahan
mlflow.log_metric("accuracy", acc)

print("Accuracy:", acc)
print("Training CI selesai")