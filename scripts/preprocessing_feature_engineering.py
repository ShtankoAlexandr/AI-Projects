import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def engineer_features(df):
    """
    Creates new features and preserves the original feature set.
    """
    df = df.copy()

    # [Added: If there are missing values, fill them with the median for numerical features]
    if df.isnull().any().any():
        print("Filling missing values with median")
        df = df.fillna(df.median(numeric_only=True))

    # Basic features
    df["Distance_to_hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2
    )
    df["Shadow_importance"] = (
        df["Hillshade_3pm"] * 0.2 + df["Hillshade_Noon"] * 0.6 + df["Hillshade_9am"] * 0.2
    )
    df["FireRoad_diff"] = df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"]

    # New features
    df["Elevation_normalized"] = df["Elevation"] / (df["Elevation"].max() + 1e-6)
    df["Slope_Aspect_cos"] = df["Slope"] * np.cos(np.radians(df["Aspect"]))
    # Process potential division by zero in Hydrology_ratio explicitly,
    # replacing Inf/NaN with 0 or another reasonable value after calculation
    df["Hydrology_ratio"] = df["Vertical_Distance_To_Hydrology"] / (df["Horizontal_Distance_To_Hydrology"] + 1e-6)
    # Replace infinite values (if Horizontal_Distance_To_Hydrology was 0) with 0
    df["Hydrology_ratio"] = df["Hydrology_ratio"].replace([np.inf, -np.inf], 0)
    # Replace NaN (if Vertical and Horizontal were 0 and Horizontal + epsilon is still very small)
    df["Hydrology_ratio"] = df["Hydrology_ratio"].fillna(0)

    # Binary features (counters)
    soil_cols = [col for col in df.columns if col.startswith("Soil_Type")]
    df["Soil_Type_count"] = df[soil_cols].sum(axis=1)
    wilderness_cols = [col for col in df.columns if col.startswith("Wilderness_Area")]
    df["Wilderness_Area_count"] = df[wilderness_cols].sum(axis=1)

    return df

def remove_original_features(df):
    """
    Removes the original features from which the composite features were derived.
    """
    features_to_drop = [
        "Elevation",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Horizontal_Distance_To_Roadways",
        # You may also consider removing Slope and Aspect if Slope_Aspect_cos is sufficient,
        # but for now we keep them as they might carry additional information.
        # Also, we keep Soil_Type_X and Wilderness_Area_X since the counters represent only part of the information.
    ]

    # Remove only the columns that exist in the DataFrame
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]

    if existing_cols_to_drop:
        print(f"Removing original features: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)
        print(f"After removing original features: {df.shape}")
    else:
        print("No original features to remove from the list.")

    return df

def load_and_preprocess_data(path_train):
    """
    Loads and preprocesses training data, including feature engineering and removal of original features.
    """
    try:
        df = pd.read_csv(path_train)
        print(f"Loaded {path_train}: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path_train} not found")

    df = engineer_features(df)
    print(f"After feature engineering: {df.shape}")

    df = remove_original_features(df)  # Adding removal step

    print("Final columns:", df.columns.tolist())

    return df

def split_and_scale(df):
    """
    Splits the data into training and testing sets without scaling the features.
    """
    if "Cover_Type" not in df.columns:
        raise ValueError("Column 'Cover_Type' not found")

    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"Data splitting: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

    return X_train, X_test, y_train, y_test

# Example usage (assuming the 'train.csv' file is present):
# path_to_train_data = 'train.csv'
# df_processed = load_and_preprocess_data(path_to_train_data)
# X_train, X_test, y_train, y_test = split_and_scale(df_processed)