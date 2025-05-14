import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
from preprocessing_feature_engineering import engineer_features
from datetime import datetime

def update_readme_after_prediction(model_name, test_acc, report):
    """
    Appends the test set prediction result to README.md.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("../README.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Test.csv Prediction Result ({now})\n")
        f.write(f"### Model: {model_name}\n")
        f.write(f"- Test.csv Accuracy: {test_acc:.4f}\n")
        f.write("\n### Classification Report:\n")
        f.write(report)
        f.write("\n")

def predict_and_save(test_path, model_path, output_path):
    """
    Loads the test data, performs predictions, and saves the results.
    """
    try:
        # Load test data
        test_df = pd.read_csv(test_path)
        print(f"test.csv shape: {test_df.shape}")
        print("Columns in test.csv:", test_df.columns.tolist())
        print("Missing values in test.csv:\n", test_df.isnull().sum())
        print("Statistics of test.csv:\n", test_df.describe())
    except FileNotFoundError:
        print(f"Error: File {test_path} not found")
        return

    # Data preprocessing: feature engineering
    test_df = engineer_features(test_df)
    print(f"After feature engineering: {test_df.shape}")
    print("Columns after feature engineering:", test_df.columns.tolist())

    # Check if the 'Cover_Type' column exists
    y_true = test_df.get("Cover_Type", None)
    if "Cover_Type" in test_df.columns:
        print("Found 'Cover_Type', removing it for predictions")
        test_df = test_df.drop(columns=["Cover_Type"])

    # Load model
    try:
        model = load(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return
    print("Expected model features:", model.feature_names_in_.tolist())

    # Check for all required features
    if not all(col in test_df.columns for col in model.feature_names_in_):
        missing_cols = [col for col in model.feature_names_in_ if col not in test_df.columns]
        raise ValueError(f"Missing features in test_df: {missing_cols}")

    # Arrange features
    test_df = test_df[model.feature_names_in_]
    print("Final features for prediction:", test_df.columns.tolist())

    # Make predictions
    y_pred = model.predict(test_df)

    # Save the result to a CSV file
    output = pd.DataFrame({"Cover_Type": y_pred})
    output.to_csv(output_path, index=False)
    print(f"✅ Predictions saved at {output_path}")

    # Diagnostics (if true labels y_true are available)
    if y_true is not None:
        test_acc = accuracy_score(y_true, y_pred)
        print(f"Test set accuracy: {test_acc:.4f}")
        if test_acc < 0.65:
            print("Warning: Test accuracy is below 0.65")
        print("\nClassification report:")
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        print("Distribution of predicted classes:")
        print(pd.Series(y_pred).value_counts(normalize=True))
        print("Distribution of true classes:")
        print(pd.Series(y_true).value_counts(normalize=True))
        
        # Update README.md
        update_readme_after_prediction(model.__class__.__name__, test_acc, report)

        return test_acc

    return None

if __name__ == "__main__":
    # Example function call
    predict_and_save('../data/test.csv', '../results/best_model.pkl', '../results/test_predictions.csv')