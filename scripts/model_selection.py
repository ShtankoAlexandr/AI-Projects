import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from imblearn.pipeline import Pipeline as imPipeline   # для моделей с SMOTE
from sklearn.pipeline import Pipeline as skPipeline  # для моделей без SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Явно указываем, что экспортируем из этого модуля
__all__ = [
    "train_models_with_gridsearch",
    "evaluate_models",
    "plot_confusion_matrix",
    "plot_learning_curve",
    "save_best_model",
    "update_readme"
]

def train_models_with_gridsearch(X_train, y_train):
    """
    Обучает модели с использованием GridSearchCV c применением SMOTE.
    Здесь выполняется подбор гиперпараметров и 5-кратная стратифицированная кросс-валидация
    на исходной (неперебалансированной) выборке. SMOTE применяется в pipeline для балансировки классов.
    """
    print("Обучение моделей с SMOTE (минимизированный набор параметров)...")
    
    models = {
    # "Gradient Boosting": (
    #     imPipeline([
    #         ("smote", SMOTE(random_state=42)),
    #         ("clf", GradientBoostingClassifier(random_state=42))
    #     ]),
    #     {
    #         "clf__n_estimators": [150], # [70,150]
    #         "clf__learning_rate": [0.2], #[0.1,02]
    #         "clf__max_depth": [15], #[15,17]
    #         "clf__min_samples_split": [2],
            
    #         "clf__min_samples_leaf": [3]
    #     }
    # ),
    
    "LogisticRegression": (
        imPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42))
        ]),
        {
            "clf__max_iter": [100], # [50,100,200,400]
            "clf__C": [0.2], # [0.01, 0.2, 0.5, 1, 10]
            "clf__solver": ["lbfgs"], #solver=["lbfgs", "liblinear", "saga"]
        }
    ),

    # "Random Forest": (
    #     imPipeline([
    #         ("smote", SMOTE(random_state=42)),
    #         ("clf", RandomForestClassifier(random_state=42))
    #     ]),
    #     {
    #         "clf__n_estimators": [200],  #[100,200]
    #         "clf__max_depth": [30], #[10,20,30]
    #         "clf__min_samples_split": [2],  #[2,5,10]
    #         "clf__min_samples_leaf": [2],#[2,3]
    #         "clf__max_features": ["sqrt"]
    #     }
    # ),

    # "KNN": (
    #     skPipeline([
    #         # No need smoote!!!
    #         ("scaler", StandardScaler()),
    #         ("pca", PCA(n_components=0.8)),
    #         ("clf", KNeighborsClassifier())
            
    #     ]),
    #     {
    #         "clf__n_neighbors": [15,20], #[5,10,30,50,100],
    #         "clf__weights": ["distance"], #["uniform", "distance"]
    #         "clf__leaf_size": [5], # [3,7,20,30,40,50]
    #         "clf__metric": ["euclidean"] #["euclidean", "manhattan"]
    #     }
    # ),

    "SVM": (
        imPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=42))
        ]),
        {
            "clf__C": [30,40], # [1, 10,20,40]
            "clf__kernel": ["rbf"], #["linear", "rbf"]
            "clf__gamma": [0.01, 0.1, "scale"] #["scale", "auto"]
        }
    ),
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
            "best_params": grid.best_params_,
            "grid_params": params
        }
        print(f"{name} - Лучший CV Score: {grid.best_score_:.4f}")
        print(f"{name} - Точность на обучающем наборе: {train_acc:.4f}")
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
    
    return results, best_model

def evaluate_models(results, X_test, y_test):
    """
    Оценивает точность и F1-score моделей на тестовой выборке.
    Возвращает два словаря: один для точностей и один для F1-score.
    """
    print("\nОценка моделей на тестовом наборе:")
    test_accuracies = {}
    test_f1_scores = {}
    for name, result in results.items():
        test_pred = result["best_model"].predict(X_test)
        acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average="weighted")
        test_accuracies[name] = acc
        test_f1_scores[name] = f1
        print(f"{name}: Точность = {acc:.4f}, F1-score = {f1:.4f}")
    return test_accuracies, test_f1_scores

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
    plt.title(f"Confusion Matrix of {model}")
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
    Обновляет файл README.md с результатами обучения моделей.
    
    Параметры:
    - results: dict, содержащий информацию о каждой модели:
        {
            'ModelName': {
                'train_accuracy': float,
                'best_score': float,
                'best_params': dict,
                'param_grid' или 'grid_params': dict (необязательно)
            }, ...
        }
    - test_accuracies: dict, содержащий точность на тестовой выборке для каждой модели.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Открытие файла для добавления новых данных
    with open("../README.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Results of training ({now})\n")
        
        for name, result in results.items():
            f.write(f"\n### {name}\n")
            f.write(f"- Train Accuracy: {result['train_accuracy']:.4f}\n")
            f.write(f"- CV Score: {result['best_score']:.4f}\n")
            
            test_acc = test_accuracies.get(name)
            if test_acc is not None:
                f.write(f"- Test Accuracy: {test_acc:.4f}\n")
            else:
                f.write("- Test Accuracy: N/A\n")

            f.write(f"- Best Params: {result['best_params']}\n")

            # Попробуем получить параметры из возможных ключей
            param_grid = result.get('grid_params') or result.get('param_grid')
            if param_grid:
                f.write(f"- Initial Grid Params:\n")
                for param, values in param_grid.items():
                    f.write(f"  - {param}: {values}\n")
            else:
                f.write("- Initial Grid Params: not available\n")


if __name__ == "__main__":
    # Импорт функций для загрузки и предобработки данных
    from preprocessing_feature_engineering import load_and_preprocess_data, split_and_scale

    # Загрузка и предобработка данных
    df = load_and_preprocess_data("../data/train.csv")
    X_train, X_test, y_train, y_test = split_and_scale(df)

    # Обучение моделей с использованием GridSearchCV (с SMOTE) и минимальным набором параметров
   
