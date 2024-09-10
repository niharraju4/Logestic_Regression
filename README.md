Certainly! Below is a more detailed documentation for your code, which you can use for your GitHub repository.

---

# Logistic Regression Implementation and Hyperparameter Tuning

This repository contains a comprehensive implementation of logistic regression for both binary and multiclass classification problems. It also includes hyperparameter tuning using GridSearchCV and RandomizedSearchCV, handling imbalanced datasets, and evaluating model performance using ROC curves and AUC scores.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Creation](#dataset-creation)
3. [Logistic Regression Implementation](#logistic-regression-implementation)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Multiclass Classification](#multiclass-classification)
6. [Handling Imbalanced Data](#handling-imbalanced-data)
7. [ROC Curve and AUC Score](#roc-curve-and-auc-score)
8. [Dependencies](#dependencies)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

Logistic regression is a widely used statistical model for binary classification problems. This repository demonstrates how to implement logistic regression, perform hyperparameter tuning, handle imbalanced datasets, and evaluate model performance using ROC curves and AUC scores.

## Dataset Creation

The dataset is created using the `make_classification` function from `sklearn.datasets`. This function generates a synthetic dataset with specified parameters.

```python
from sklearn.datasets import make_classification

# Creating a dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=15)
```

## Logistic Regression Implementation

The logistic regression model is implemented using the `LogisticRegression` class from `sklearn.linear_model`. The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Implementing logistic regression
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred = log.predict(X_test)

# Evaluating the model
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Hyperparameter Tuning

Hyperparameter tuning is performed using `GridSearchCV` and `RandomizedSearchCV` from `sklearn.model_selection`. The parameter grid includes different values for `penalty`, `C`, and `solver`.

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Defining the parameter grid
params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [100, 10, 1.0, 0.1, 0.01],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# StratifiedKFold for cross-validation
cv = StratifiedKFold()

# GridSearchCV
grid = GridSearchCV(estimator=LogisticRegression(), param_grid=params, scoring='accuracy', cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# Evaluating the model
y_pred = grid.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

# RandomizedSearchCV
randomcv = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=params, cv=5, scoring='accuracy')
randomcv.fit(X_train, y_train)
print("Best Parameters:", randomcv.best_params_)
print("Best Score:", randomcv.best_score_)

# Evaluating the model
y_pred = randomcv.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Multiclass Classification

Logistic regression can also be used for multiclass classification problems. The dataset is created with multiple classes, and the model is trained and evaluated similarly.

```python
# Creating a multiclass dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=15)

# Implementing logistic regression for multiclass classification
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)

# Evaluating the model
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Handling Imbalanced Data

For imbalanced datasets, the `class_weight` parameter can be used to handle the imbalance. The parameter grid is updated to include different class weights.

```python
from collections import Counter

# Creating an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=2, n_clusters_per_class=1, n_redundant=0, weights=[0.99], random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

# Defining the parameter grid with class weights
class_weight = [{0: w, 1: y} for w in [1, 10, 50, 100] for y in [1, 10, 50, 100]]
params = [
    {'penalty': ['l2'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['newton-cg', 'lbfgs', 'sag'], 'class_weight': class_weight},
    {'penalty': ['l1'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['liblinear'], 'class_weight': class_weight},
    {'penalty': ['elasticnet'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver': ['saga'], 'class_weight': class_weight, 'l1_ratio': [0.5]}
]

# GridSearchCV for imbalanced data
grid = GridSearchCV(estimator=LogisticRegression(), param_grid=params, scoring='accuracy', cv=cv)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# Evaluating the model
y_pred = grid.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy:", score)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## ROC Curve and AUC Score

The ROC curve and AUC score are used to evaluate the performance of the logistic regression model. The ROC curve is plotted, and the AUC score is calculated.

```python
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

# Creating a binary classification dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=10)

# Creating a dummy model with default 0 as output
dummy_model_prob = [0 for _ in range(len(y_test))]

# Implementing logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)
model_prob = model.predict_proba(X_test)[:, 1]

# Calculating the AUC score
dummy_model_auc = roc_auc_score(y_test, dummy_model_prob)
model_auc = roc_auc_score(y_test, model_prob)
print("Dummy Model AUC:", dummy_model_auc)
print("Logistic Model AUC:", model_auc)

# Calculating ROC curves
dummy_fpr, dummy_tpr, _ = roc_curve(y_test, dummy_model_prob)
model_fpr, model_tpr, thresholds = roc_curve(y_test, model_prob)

# Plotting the ROC curve
pyplot.plot(dummy_fpr, dummy_tpr, linestyle='--', label='Dummy Model')
pyplot.plot(model_fpr, model_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

# Plotting the ROC curve with thresholds
fig = pyplot.figure(figsize=(20, 50))
pyplot.plot(dummy_fpr, dummy_tpr, linestyle='--', label='Dummy Model')
pyplot.plot(model_fpr, model_tpr, marker='.', label='Logistic')
ax = fig.add_subplot(111)
for xyz in zip(model_fpr, model_tpr, thresholds):
    ax.annotate('%s' % np.round(xyz[2], 2), xy=(xyz[0], xyz[1]))
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
```

## Dependencies

The following Python libraries are required to run the code:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

To use the code, clone the repository and run the Python script. Make sure to have the required dependencies installed.

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
python your_script.py
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This documentation provides a comprehensive overview of the code and its functionality. You can customize it further based on your specific requirements.
