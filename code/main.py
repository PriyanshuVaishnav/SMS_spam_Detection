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

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Trained model or vectorizer not found. Please run training first.")

    # Load trained model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Convert input to DataFrame
    test_df = pd.DataFrame({"Processed_sms_message": [msg]})

    # Preprocessing
    txtp = TextPreprocessor(n_jobs=-1)
    test_df["Processed_sms_message"] = txtp.transform(test_df["Processed_sms_message"])

    # Vectorization
    tfidf_features = vectorizer.transform(test_df["Processed_sms_message"])
    test_features_df = pd.DataFrame(tfidf_features.toarray(), columns=vectorizer.get_feature_names_out())

    # Add additional features
    fe = FeatureEngineering(test_df)
    engineered_features = fe.add_more_features(test_df)

    # Drop message column
    engineered_features.drop(columns=["Processed_sms_message"], inplace=True, errors="ignore")

    # Combine TF-IDF + Engineered features
    final_test_input = pd.concat([test_features_df.reset_index(drop=True), engineered_features.reset_index(drop=True)], axis=1)

    # Predict
    prediction = model.predict(final_test_input)
    return prediction[0]


# ========================================
#              MAIN TRAINING
# ========================================
def main():
    data_path = Path("data/SMSSpamCollection")  # <-- update if needed

    # Step 1: Load Data
    df = DataUtils.get_data(str(data_path))

    # Step 2: EDA
    data_analysis = DataAnalysis(df)
    data_analysis.explore_data_visualization(show_word_cloud_with_specific_labels=True)

    # Step 3: Text Preprocessing
    txtp = TextPreprocessor(n_jobs=-1)
    df["Processed_sms_message"] = txtp.transform(df["sms_message"])

    # Step 4: Feature Engineering
    fe = FeatureEngineering(df)
    fe.map_labels()
    df = fe.add_more_features(df)
    df["Number_of_characters_per_word"].fillna(df["Number_of_characters_per_word"].mean(), inplace=True)

    # Step 5: Data Split
    data_dev = DatasetDevelopment(df)
    x_train, x_test, y_train, y_test = data_dev.divide_your_data()
    x_train.drop(["sms_message"], axis=1, inplace=True)
    x_test.drop(["sms_message"], axis=1, inplace=True)

    # Step 6: Extract TF-IDF Features
    Final_Training_data, Final_Test = fe.extract_features(x_train, x_test)

    # Step 7: Train Model
    trainer = ModelTraining(Final_Training_data, y_train)
    model = trainer.Xgboost_model(fine_tuning=False)

    # Step 8: Evaluate
    evaluator = EvaluateModel(Final_Test, y_test, model)
    evaluator.evaluate_model()
    evaluator.plot_confusion_matrix(y_test, model.predict(Final_Test))
    evaluator.plot_roc_curve(y_test, model.predict_proba(Final_Test)[:, 1])

    # Step 9: Save model and vectorizer
    Path("saved_models").mkdir(exist_ok=True)
    joblib.dump(model, "saved_models/XGboost_model.pkl")
    joblib.dump(fe.vectorizer, "saved_models/vectorizer.pkl")

    print("[INFO] Training completed and model saved!")


if __name__ == "__main__":
    # First run the training
    main()

    # Then test prediction
    test_msg = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
    prediction = predict(test_msg)
    print(f"[PREDICTION]: '{test_msg}' => {'Spam' if prediction else 'Ham'}")
