import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # Импорт SVC для модели SVM
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_models_with_gridsearch(X_train, y_train):
    """
    Обучает модели с использованием GridSearchCV.
    Здесь выполняется подбор гиперпараметров и 5-кратная стратифицированная кросс-валидация
    на исходной (неперебалансированной) выборке.
    """
    print("Обучение моделей (без использования SMOTE)...")
    
    # Определение моделей с гиперпараметрическими сетками, включая SVM
    models = {
        "Gradient Boosting": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=42))
            ]),
            {
                "clf__n_estimators": [5, 10, 50],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__max_depth": [2, 3, 4]
            }
        ),
        "RandomForest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
            ]),
            {
                "clf__n_estimators": [10, 50, 100],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 10, 50]
            }
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            {
                "clf__n_neighbors": [5, 10, 20]
            }
        ),
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42))
            ]),
            {
                "clf__C": [0.01, 0.05, 0.1, 1]
            }
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=42, class_weight="balanced"))
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["linear", "rbf"],
                "clf__gamma": ["scale", "auto"]
            }
        )
    }
    
    best_score = 0
    best_model = None
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, (model, params) in models.items():
        print(f"\nОбучение модели: {name}")
        grid = GridSearchCV(model, params, cv=kfold, n_jobs=-1, scoring="balanced_accuracy")
        grid.fit(X_train, y_train)
        train_pred = grid.best_estimator_.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        results[name] = {
            "best_model": grid.best_estimator_,
            "best_score": grid.best_score_,
            "train_accuracy": train_acc,
        }
        print(f"{name} - Лучший CV Score: {grid.best_score_:.4f}")
        print(f"{name} - Точность на обучающем наборе: {train_acc:.4f}")
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
    
    return results, best_model

def evaluate_models(results, X_test, y_test):
    """
    Оценивает точность моделей на тестовой выборке.
    """
    print("\nОценка моделей на тестовом наборе:")
    test_accuracies = {}
    for name, result in results.items():
        test_pred = result["best_model"].predict(X_test)
        score = accuracy_score(y_test, test_pred)
        test_accuracies[name] = score
        print(f"{name}: Точность = {score:.4f}")
    return test_accuracies

def plot_confusion_matrix(model, X_test, y_test):
    """
    Строит confusion matrix для модели и возвращает DataFrame
    с индексами 'True label' и столбцами 'Predicted label'.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True {i}" for i in range(cm.shape[0])],
                         columns=[f"Pred {i}" for i in range(cm.shape[1])])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    return cm_df

def plot_learning_curve(model, X_train, y_train, cv=5):
    """
    Строит график learning curve для модели с использованием заданного количества фолдов.
    """
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train,
        cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1, 10),
        scoring="balanced_accuracy"
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training Accuracy")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="green", label="Validation Accuracy")
    plt.title("Learning Curve")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Balanced Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def save_best_model(best_model):
    """
    Сохраняет лучшую модель в файл ../results/best_model.pkl.
    """
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dump(best_model, os.path.join(results_dir, "best_model.pkl"))
    print("✅ Модель сохранена в '../results/best_model.pkl'")

def update_readme(results, test_accuracies):
    """
    Обновляет файл README.md с результатами обучения.
    """
    with open("../README.md", "a") as f:
        f.write("\n## Результаты обучения\n")
        for name, result in results.items():
            f.write(
                f"- {name}: Train Accuracy = {result['train_accuracy']:.4f}, "
                f"CV Score = {result['best_score']:.4f}, "
                f"Test Accuracy = {test_accuracies.get(name, 'N/A'):.4f}\n"
            )

if __name__ == "__main__":
    # Импорт функций для загрузки и предобработки данных
    from preprocessing_feature_engineering import load_and_preprocess_data, split_and_scale
    
    # Загрузка и предобработка данных
    df = load_and_preprocess_data("../data/train.csv")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    
    # Обучение моделей с использованием GridSearchCV (без SMOTE)
    results, best_model = train_models_with_gridsearch(X_train, y_train)
    test_accuracies = evaluate_models(results, X_test, y_test)
    
    # Построение и отображение confusion matrix для лучшей модели
    cm_df = plot_confusion_matrix(best_model, X_test, y_test)
    print("Confusion Matrix:\n", cm_df)
    
    # Построение графика learning curve для лучшей модели
    plot_learning_curve(best_model, X_train, y_train, cv=5)
    
    # Сохранение лучшей модели и обновление README
    save_best_model(best_model)
    update_readme(results, test_accuracies)