# train_ml.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load your preprocessed dataset
# Make sure it has two columns: 'text' and 'label'
df = pd.read_csv("datasets_folder/train.csv")  # change path if needed

# 2️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 3️⃣ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5️⃣ Train SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

print("🔹 SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# 6️⃣ Save models and vectorizer
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(svm_model, "svm_model.pkl")
print("✅ Models saved successfully!")
