import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
from preprocessing_feature_engineering import engineer_features
from datetime import datetime

def update_readme_after_prediction(model_name, test_acc, report):
    """
    Добавляет в README.md результат предсказания модели на тестовом наборе.
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
    Загружает тестовые данные, выполняет предсказания и сохраняет результаты.
    """
    try:
        # Загрузка тестовых данных
        test_df = pd.read_csv(test_path)
        print(f"test.csv shape: {test_df.shape}")
        print("Столбцы в test.csv:", test_df.columns.tolist())
        print("Пропуски в test.csv:\n", test_df.isnull().sum())
        print("Статистика test.csv:\n", test_df.describe())
    except FileNotFoundError:
        print(f"Ошибка: Файл {test_path} не найден")
        return

    # Предобработка данных: создание новых признаков
    test_df = engineer_features(test_df)
    print(f"После feature engineering: {test_df.shape}")
    print("Столбцы после feature engineering:", test_df.columns.tolist())

    # Проверка наличия столбца "Cover_Type"
    y_true = test_df.get("Cover_Type", None)
    if "Cover_Type" in test_df.columns:
        print("Найден Cover_Type, удаляем его для предсказаний")
        test_df = test_df.drop(columns=["Cover_Type"])

    # Загрузка модели
    try:
        model = load(model_path)
    except FileNotFoundError:
        print(f"Ошибка: Модель не найдена по пути {model_path}")
        return
    print("Ожидаемые признаки модели:", model.feature_names_in_.tolist())

    # Проверка наличия всех необходимых признаков
    if not all(col in test_df.columns for col in model.feature_names_in_):
        missing_cols = [col for col in model.feature_names_in_ if col not in test_df.columns]
        raise ValueError(f"Отсутствуют признаки в test_df: {missing_cols}")

    # Упорядочивание признаков
    test_df = test_df[model.feature_names_in_]
    print("Финальные признаки для предсказания:", test_df.columns.tolist())

    # Выполнение предсказаний
    y_pred = model.predict(test_df)

    # Сохранение результата в CSV-файл
    output = pd.DataFrame({"Cover_Type": y_pred})
    output.to_csv(output_path, index=False)
    print(f"✅ Предсказания сохранены в {output_path}")

    # Диагностика (если доступны истинные метки y_true)
    if y_true is not None:
        test_acc = accuracy_score(y_true, y_pred)
        print(f"Точность на тестовом наборе: {test_acc:.4f}")
        if test_acc < 0.65:
            print("Предупреждение: Test accuracy ниже 0.65")
        print("\nОтчет по классификации:")
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        print("Распределение предсказанных классов:")
        print(pd.Series(y_pred).value_counts(normalize=True))
        print("Распределение истинных классов:")
        print(pd.Series(y_true).value_counts(normalize=True))
        
        # Обновление README.md
        update_readme_after_prediction(model.__class__.__name__, test_acc, report)

        return test_acc

    return None

if __name__ == "__main__":
    # Пример вызова функции 
    predict_and_save('../data/test.csv', '../results/best_model.pkl', '../results/test_predictions.csv')
