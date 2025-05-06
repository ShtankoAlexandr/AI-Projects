import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Создает новые признаки и сохраняет исходный набор признаков.
    """
    df = df.copy()

    # [Добавлено: Если есть пропуски, заполняем медианой для числовых признаков]
    if df.isnull().any().any():
        print("Заполняем пропуски медианой")
        df = df.fillna(df.median(numeric_only=True))

    # Основные признаки
    df["Distance_to_hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2
    )
    df["Shadow_importance"] = (
        df["Hillshade_3pm"] * 0.2 + df["Hillshade_Noon"] * 0.6 + df["Hillshade_9am"] * 0.2
    )
    df["FireRoad_diff"] = df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Roadways"]

    # Новые признаки
    df["Elevation_normalized"] = df["Elevation"] / (df["Elevation"].max() + 1e-6)
    df["Slope_Aspect_cos"] = df["Slope"] * np.cos(np.radians(df["Aspect"]))
    df["Hydrology_ratio"] = (
        df["Vertical_Distance_To_Hydrology"] / (df["Horizontal_Distance_To_Hydrology"] + 1e-6)
    )

    # Бинарные признаки
    soil_cols = [col for col in df.columns if col.startswith("Soil_Type")]
    df["Soil_Type_count"] = df[soil_cols].sum(axis=1)
    wilderness_cols = [col for col in df.columns if col.startswith("Wilderness_Area")]
    df["Wilderness_Area_count"] = df[wilderness_cols].sum(axis=1)

    return df

def load_and_preprocess_data(path_train):
    """
    Загружает и предобрабатывает тренировочные данные.
    """
    try:
        df = pd.read_csv(path_train)
        print(f"Загружено {path_train}: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path_train} не найден")

    df = engineer_features(df)
    print(f"После feature engineering: {df.shape}")
    print("Столбцы:", df.columns.tolist())

    return df

def split_and_scale(df):
    """
    Разделяет данные на обучающую и тестовую выборки и масштабирует признаки.
    """
    if "Cover_Type" not in df.columns:
        raise ValueError("Отсутствует столбец 'Cover_Type'")

    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test, scaler