# 🌲 Forest Cover Type Classification

This project aims to develop machine learning models to accurately classify forest cover types based on cartographic variables. The goal is to generate predictions on unseen data and interpret the results to support stakeholders in a forest conservation agency. All processes are documented to ensure transparency, reproducibility, and stakeholder communication.

---

## 📦 Dataset

The dataset is provided in three files:

- `train.csv`: Main dataset used for model training and validation.
- `test.csv`: Dataset used for model testing and validation.
- `covtype.info`: Metadata file describing all features and their meanings.

---

## 🧪 Project Goals

- Build interpretable and accurate classification models.
- Avoid overfitting by proper validation and separation of final test data.
- Clearly document the pipeline for stakeholders.
- Meet required performance metrics:
  - **Train accuracy < 0.98**
  - **Test accuracy > 0.65**

---

## 📊 Workflow Overview

### 1. Exploratory Data Analysis (EDA) & Feature Engineering

Performed in a Jupyter notebook (`notebook/EDA.ipynb`), EDA explores feature distributions, class balance, and potential new feature creation. Example engineered features:

- `Euclidean_Distance_To_Hydrology = sqrt(Horizontal_Distance_To_Hydrology² + Vertical_Distance_To_Hydrology²)`
- `Firepoints_vs_Roadways = Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways`

### 2. Model Selection (`scripts/model_selection.py`)

A robust evaluation pipeline is implemented:

- **Train/Validation Split**: Stratified 5-fold cross-validation
- **Model Candidates**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting

- **Grid Search** over hyperparameters
- **Learning curve** plotted for the best model
- **Confusion Matrix** presented as labeled DataFrame
- **Best model saved** as `results/best_model.pkl`

### 3. Final Prediction (`scripts/predict.py`)

On the last day, after test data becomes available:

- Load `best_model.pkl`
- Predict test labels
- Evaluate and report accuracy
- Save results to `results/test_predictions.csv`

---

## 🗂️ Project Structure
project/
│ README.md
│ requirements.txt
│
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── covtype.info
│
├── notebook/
│ └── EDA.ipynb
│
├── scripts/
│ ├── preprocessing_feature_engineering.py
│ ├── model_selection.py
│ └── predict.py
│
└── results/
├── plots/
├── best_model.pkl
└── test_predictions.csv


---

## 🔧 How to Run

### 1. Clone and Setup Environment

```bash
git clone https://github.com/ShtankoAlexandr/AI-Projects
cd forest-prediction
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Step 1: Feature Engineering
python scripts/preprocessing_feature_engineering.py

# Step 2: Model Selection
python scripts/model_selection.py

# Step 3: Final Prediction
python scripts/predict.py

👤 Author
Oleksandr (Alex) Shtanko
Senior Product Manager | Machine Learning Engineer
📍 Denham, Uxbridge, London
📧 al222ex@gmail.com
🔗 www.linkedin.com/in/oleksandr-shtanko









## Results of training (2025-05-14 15:05:45)

### Gradient Boosting
- Train Accuracy: 0.9480
- CV Score: 0.8798
- Test Accuracy: 0.9088
- Best Params: {'clf__learning_rate': 0.1, 'clf__max_depth': 5, 'clf__min_samples_leaf': 5, 'clf__min_samples_split': 4, 'clf__n_estimators': 180}
- Initial Grid Params:
  - clf__n_estimators: [180]
  - clf__learning_rate: [0.1]
  - clf__max_depth: [5]
  - clf__min_samples_split: [4]
  - clf__min_samples_leaf: [5]

### LogisticRegression
- Train Accuracy: 0.6992
- CV Score: 0.7149
- Test Accuracy: 0.7071
- Best Params: {'clf__C': 0.2, 'clf__max_iter': 100, 'clf__solver': 'lbfgs'}
- Initial Grid Params:
  - clf__max_iter: [100]
  - clf__C: [0.1, 0.2]
  - clf__solver: ['lbfgs', 'saga']

## Test.csv Prediction Result (2025-05-14 15:06:03)
### Model: Pipeline
- Test.csv Accuracy: 0.6512

### Classification Report:
              precision    recall  f1-score   support

           1       0.68      0.70      0.69    201684
           2       0.80      0.47      0.59    254393
           3       0.70      0.81      0.75     33594
           4       0.21      0.97      0.34       587
           5       0.11      0.93      0.19      7077
           6       0.42      0.79      0.55     15207
           7       0.42      0.95      0.59     18350

    accuracy                           0.61    530892
   macro avg       0.48      0.80      0.53    530892
weighted avg       0.71      0.61      0.63    530892


## Results of training (2025-05-14 16:42:37)

### Gradient Boosting
- Train Accuracy: 0.9754
- CV Score: 0.8831
- Test Accuracy: 0.9174
- Best Params: {'clf__learning_rate': 0.2, 'clf__max_depth': 5, 'clf__min_samples_leaf': 3, 'clf__min_samples_split': 4, 'clf__n_estimators': 180}
- Initial Grid Params:
  - clf__n_estimators: [180]
  - clf__learning_rate: [0.2]
  - clf__max_depth: [5]
  - clf__min_samples_split: [4]
  - clf__min_samples_leaf: [3]

### LogisticRegression
- Train Accuracy: 0.6992
- CV Score: 0.7149
- Test Accuracy: 0.7071
- Best Params: {'clf__C': 0.2, 'clf__max_iter': 100, 'clf__solver': 'lbfgs'}
- Initial Grid Params:
  - clf__max_iter: [100]
  - clf__C: [0.1, 0.2]
  - clf__solver: ['lbfgs', 'saga']

### Random Forest
- Train Accuracy: 0.9827
- CV Score: 0.8920
- Test Accuracy: 0.9282
- Best Params: {'clf__max_depth': 28, 'clf__max_features': 0.5, 'clf__min_samples_leaf': 3, 'clf__min_samples_split': 3, 'clf__n_estimators': 150}
- Initial Grid Params:
  - clf__n_estimators: [150]
  - clf__max_depth: [28]
  - clf__min_samples_split: [3]
  - clf__min_samples_leaf: [3]
  - clf__max_features: ['sqrt', 0.5]

### KNN
- Train Accuracy: 1.0000
- CV Score: 0.8013
- Test Accuracy: 0.8372
- Best Params: {'clf__leaf_size': 3, 'clf__metric': 'manhattan', 'clf__n_neighbors': 50, 'clf__weights': 'distance'}
- Initial Grid Params:
  - clf__n_neighbors: [50]
  - clf__weights: ['distance']
  - clf__leaf_size: [3]
  - clf__metric: ['manhattan']

### SVM
- Train Accuracy: 0.9242
- CV Score: 0.8500
- Test Accuracy: 0.8887
- Best Params: {'clf__C': 40, 'clf__gamma': 0.1, 'clf__kernel': 'rbf'}
- Initial Grid Params:
  - clf__C: [40]
  - clf__kernel: ['rbf']
  - clf__gamma: [0.01, 0.1, 'scale']

## Test.csv Prediction Result (2025-05-14 16:43:01)
### Model: Pipeline
- Test.csv Accuracy: 0.6648

### Classification Report:
              precision    recall  f1-score   support

           1       0.70      0.73      0.71    201684
           2       0.83      0.49      0.62    254393
           3       0.69      0.84      0.76     33594
           4       0.20      0.97      0.34       587
           5       0.11      0.93      0.20      7077
           6       0.42      0.79      0.55     15207
           7       0.45      0.96      0.62     18350

    accuracy                           0.63    530892
   macro avg       0.49      0.82      0.54    530892
weighted avg       0.74      0.63      0.65    530892

