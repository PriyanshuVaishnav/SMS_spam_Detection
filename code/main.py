from typing import Text
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from run_prepare_data import DatasetDevelopment, DataUtils
from data_analysis import DataAnalysis
from text_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate import EvaluateModel


# ========================================
#            PREDICTION FUNCTION
# ========================================
def predict(msg: str):
    model_path = Path("saved_models/XGboost_model.pkl")
    vectorizer_path = Path("saved_models/vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Vectorize input
    test_vector = vectorizer.transform([msg])
    data = pd.DataFrame(test_vector.toarray(), columns=vectorizer.get_feature_names_out())

    # Preprocessing
    data["Processed_sms_message"] = msg
    txtp = TextPreprocessor(n_jobs=-1)
    data["Processed_sms_message"] = txtp.transform(data["Processed_sms_message"])

    # Feature Engineering
    feature_engineering = FeatureEngineering(data)
    df_new_features = feature_engineering.add_more_features(data)
    df_new_features.drop(["Processed_sms_message"], axis=1, inplace=True)

    # Prediction
    output = model.predict(df_new_features)
    return output[0]


# ========================================
#              MAIN TRAINING
# ========================================
def main():
    data_path = Path("data/SMSSpamCollection")  # Make sure this path exists on cloud

    df = DataUtils.get_data(str(data_path))

    # Data Analysis
    data_analysis = DataAnalysis(df)
    data_analysis.explore_data_visualization(show_word_cloud_with_specific_labels=True)

    # Text Processing
    txtp = TextPreprocessor(n_jobs=-1)
    df["Processed_sms_message"] = txtp.transform(df["sms_message"])

    # Feature Engineering
    feature_engineering = FeatureEngineering(df)
    feature_engineering.map_labels()
    df = feature_engineering.add_more_features(df)
    df["Number_of_characters_per_word"].fillna(df["Number_of_characters_per_word"].mean(), inplace=True)

    # Train/Test Split
    data_dev = DatasetDevelopment(df)
    x_train, x_test, y_train, y_test = data_dev.divide_your_data()
    x_train.drop(["sms_message"], axis=1, inplace=True)
    x_test.drop(["sms_message"], axis=1, inplace=True)

    # Extract Features
    Final_Training_data, Final_Test = feature_engineering.extract_features(x_train, x_test)

    # Train Model
    model_train = ModelTraining(Final_Training_data, y_train)
    model = model_train.Xgboost_model(fine_tuning=False)

    # Evaluate
    evaluate = EvaluateModel(Final_Test, y_test, model)
    evaluate.evaluate_model()
    evaluate.plot_confusion_matrix(y_test, model.predict(Final_Test))
    evaluate.plot_roc_curve(y_test, model.predict_proba(Final_Test)[:, 1])

    # Save model & vectorizer
    Path("saved_models").mkdir(exist_ok=True)
    joblib.dump(model, "saved_models/XGboost_model.pkl")
    joblib.dump(feature_engineering.vectorizer, "saved_models/vectorizer.pkl")


if __name__ == "__main__":
    # Test prediction
    print(
        predict("Congratulations! You've won a $1000 Walmart gift card. Click here to claim.")
    )

