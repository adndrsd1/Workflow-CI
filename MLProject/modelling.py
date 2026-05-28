import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hubungkan path file dataset secara aman menggunakan relative path dinamis
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "../Titanic-Dataset_preprocessing.csv"))

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# HAPUS BARIS: mlflow.set_tracking_uri(...) 
# HAPUS BARIS: mlflow.set_experiment(...)
# Biarkan parameter tracking diatur secara otomatis oleh file manifes MLproject

# Aktifkan perekaman metrik otomatis dari scikit-learn
mlflow.sklearn.autolog(log_models=True)

# ========================================================
# PROSES TRAINING & EVALUASI
# ========================================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Tarik proses pelatihan (Metrik akurasi otomatis direkam oleh autolog)
model.fit(X_train, y_train)

# Prediksi komponen test
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("====================================")
print("Accuracy via MLflow CLI:", acc)
print("Training selesai dengan sukses!")
print("====================================")