# evaluate_models.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Test Data ===
test_data = pd.read_csv("datasets_folder/test.csv")

# === Separate Features and Labels ===
X_test = test_data["text"]
y_test = test_data["label"]

# === Evaluate ML Model ===
print("\n================= MACHINE LEARNING MODEL =================")

ml_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

X_test_vectorized = vectorizer.transform(X_test)
y_pred_ml = ml_model.predict(X_test_vectorized)

# Metrics
acc_ml = accuracy_score(y_test, y_pred_ml)
f1_ml = f1_score(y_test, y_pred_ml)
prec_ml = precision_score(y_test, y_pred_ml)
rec_ml = recall_score(y_test, y_pred_ml)

print(f"Accuracy: {acc_ml:.4f}")
print(f"F1 Score: {f1_ml:.4f}")
print(f"Precision: {prec_ml:.4f}")
print(f"Recall: {rec_ml:.4f}")

# === Confusion Matrix (formatted table) ===

cm_ml = confusion_matrix(y_test, y_pred_ml)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'])
plt.title("Confusion Matrix - ML Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
