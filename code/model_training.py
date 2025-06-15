from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


class ModelTraining:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def logistic_regression_model(self):
        print("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(self.x_train, self.y_train)
        print("Training completed.")
        return model

    def Xgboost_model(self, fine_tuning=True):
        if fine_tuning:
            print("Hyperparameter tuning for XGBoost...")
            params = {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [2, 4, 6, 8],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                "subsample": [0.5, 0.7, 1.0],
                "colsample_bytree": [0.5, 0.7, 1.0],
                "min_child_weight": [1, 2, 3],
                "gamma": [0, 0.1, 0.3],
            }
            model = RandomizedSearchCV(XGBClassifier(), params, cv=5, n_jobs=-1)
            model.fit(self.x_train, self.y_train)
            print("XGBoost tuning complete.")
            return model
        else:
            print("Training XGBoost model...")
            model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
            model.fit(self.x_train, self.y_train)
            print("XGBoost training completed.")
            return model

    def svm_model(self, fine_tuning=True):
        if fine_tuning:
            print("Hyperparameter tuning for SVM...")
            params = {
                "C": [0.1, 0.5, 1.0],
                "gamma": ['scale', 'auto'],
                "kernel": ['rbf', 'linear'],
            }
            model = RandomizedSearchCV(SVC(), params, cv=5, n_jobs=-1)
            model.fit(self.x_train, self.y_train)
            print("SVM tuning complete.")
            return model
        else:
            print("Training SVM model...")
            model = SVC(C=1.0, gamma='scale', kernel='rbf')
            model.fit(self.x_train, self.y_train)
            print("SVM training completed.")
            return model

    def random_forest_model(self, fine_tuning=True):
        if fine_tuning:
            print("Hyperparameter tuning for Random Forest...")
            params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
            model = RandomizedSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1)
            model.fit(self.x_train, self.y_train)
            print("Random Forest tuning complete.")
            return model
        else:
            print("Training Random Forest model...")
            model = RandomForestClassifier(n_estimators=100)
            model.fit(self.x_train, self.y_train)
            print("Random Forest training completed.")
            return model

    def Naive_Bayes(self):
        print("Training Naive Bayes model...")
        model = MultinomialNB()
        model.fit(self.x_train, self.y_train)
        print("Naive Bayes training completed.")
        return model

    def KNN_model(self, fine_tuning=True):
        if fine_tuning:
            print("Hyperparameter tuning for KNN...")
            params = {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'ball_tree', 'kd_tree'],
                "leaf_size": [10, 20, 30],
                "p": [1, 2],
            }
            model = RandomizedSearchCV(KNeighborsClassifier(), params, cv=5, n_jobs=-1)
            model.fit(self.x_train, self.y_train)
            print("KNN tuning complete.")
            return model
        else:
            print("Training KNN model...")
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(self.x_train, self.y_train)
            print("KNN training completed.")
            return model
# train_model.py

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load your dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_num"], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
