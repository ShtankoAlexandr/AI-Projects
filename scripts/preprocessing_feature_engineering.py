import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
    # Обрабатываем возможное деление на ноль в Hydrology_ratio более явно,
    # заменяя Inf/NaN на 0 или другое разумное значение после вычисления
    df["Hydrology_ratio"] = df["Vertical_Distance_To_Hydrology"] / (df["Horizontal_Distance_To_Hydrology"] + 1e-6)
    # Заменяем бесконечные значения (если Horizontal_Distance_To_Hydrology было 0) на 0
    df["Hydrology_ratio"] = df["Hydrology_ratio"].replace([np.inf, -np.inf], 0)
    # Заменяем NaN (если Vertical и Horizontal были 0 и Horizontal + epsilon все равно очень мало)
    df["Hydrology_ratio"] = df["Hydrology_ratio"].fillna(0)


    # Бинарные признаки (счетчики)
    soil_cols = [col for col in df.columns if col.startswith("Soil_Type")]
    df["Soil_Type_count"] = df[soil_cols].sum(axis=1)
    wilderness_cols = [col for col in df.columns if col.startswith("Wilderness_Area")]
    df["Wilderness_Area_count"] = df[wilderness_cols].sum(axis=1)

    return df

def remove_original_features(df):
    """
    Удаляет исходные признаки, из которых были получены новые композитные признаки.
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
        # Можно также рассмотреть удаление Slope и Aspect, если Slope_Aspect_cos достаточен,
        # но пока оставим их, т.к. они могут нести дополнительную информацию.
        # Также оставим Soil_Type_X и Wilderness_Area_X, т.к. счетчики - это только часть информации.
    ]

    # Удаляем только те столбцы, которые существуют в DataFrame
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]

    if existing_cols_to_drop:
        print(f"Удаляем исходные признаки: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)
        print(f"После удаления исходных признаков: {df.shape}")
    else:
        print("Нет исходных признаков для удаления из списка.")

    return df


def load_and_preprocess_data(path_train):
    """
    Загружает и предобрабатывает тренировочные данные, включая инженерию и удаление исходных признаков.
    """
    try:
        df = pd.read_csv(path_train)
        print(f"Загружено {path_train}: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path_train} не найден")

    df = engineer_features(df)
    print(f"После feature engineering: {df.shape}")

    df = remove_original_features(df) # Добавляем шаг удаления

    print("Финальные столбцы:", df.columns.tolist())

    return df

def split_and_scale(df):
    """
    Разделяет данные на обучающую и тестовую выборки без масштабирования признаков.
    """
    if "Cover_Type" not in df.columns:
        raise ValueError("Отсутствует столбец 'Cover_Type'")

    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"Разделение данных: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

    return X_train, X_test, y_train, y_test

# Пример использования (предполагается наличие файла 'train.csv'):
# path_to_train_data = 'train.csv'
# df_processed = load_and_preprocess_data(path_to_train_data)
# X_train, X_test, y_train, y_test = split_and_scale(df_processed)