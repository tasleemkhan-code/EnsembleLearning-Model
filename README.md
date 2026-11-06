# **Ensemble Learning on the Iris Dataset**

A hands-on comparison of the main ensemble learning techniques, **Stacking**, **Bagging (Random Forest)** and **Boosting (AdaBoost, Gradient Boosting, XGBoost)**, applied to the classic Iris flower classification problem using scikit-learn and XGBoost.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ensemble-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

---

# **Overview**

Ensemble methods combine several models to get better predictive performance than a single model. This project implements five ensemble classifiers on the same train/test split so their results can be compared fairly.

| Family | Models implemented |
|---|---|
| **Stacking** | `StackingClassifier` with Decision Tree, SVM (RBF) and Logistic Regression as base learners, and Logistic Regression as the meta-learner |
| **Bagging** | `RandomForestClassifier` |
| **Boosting** | `AdaBoostClassifier`, `GradientBoostingClassifier`, `XGBClassifier` |

# **Dataset**

- **Iris dataset**, loaded directly with `seaborn.load_dataset('iris')`
- 150 samples, 4 numeric features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Target: `species` (setosa, versicolor, virginica), encoded with `LabelEncoder`

# **Methodology**

1. **Load data** from seaborn and separate features (`x`) and target (`y`).
2. **Encode labels** with `LabelEncoder`.
3. **Split data** 80/20 into train and test sets with `random_state=42` and `stratify=y`, so class proportions are preserved.
4. **Train each ensemble** on the same training set.
5. **Evaluate** every model with accuracy on the held-out test set (30 samples).

## **Model Configuration**

| Model | Key settings |
|---|---|
| Stacking | Base learners: `DecisionTreeClassifier`, `SVC(kernel='rbf', probability=True)`, `LogisticRegression`. Meta-learner: `LogisticRegression(max_iter=1000)`. `cv=5` |
| Random Forest | `n_estimators=100`, `max_depth=None` |
| AdaBoost | `n_estimators=100` |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1` |
| XGBoost | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `eval_metric='mlogloss'` |

# **Results**

All models were evaluated on the same stratified test set (30 samples).

| Model | Technique | Test Accuracy |
|---|---|---|
| Stacking Classifier | Stacking | **96.67%** |
| Gradient Boosting | Boosting | **96.67%** |
| AdaBoost | Boosting | 93.33% |
| XGBoost | Boosting | 93.33% |
| Random Forest | Bagging | 90.00% |

## **Key Observations**

- Stacking and Gradient Boosting tied for the best accuracy at 96.67% (29 of 30 correct).
- Stacking benefits from combining models with different inductive biases (tree-based, kernel-based and linear).
- With only 30 test samples, one prediction equals about 3.33 percentage points, so small differences between models are **not statistically significant**. For a more reliable comparison, use k-fold cross-validation (see Future Improvements).

# **Getting Started**

## **Prerequisites**

- Python 3.9+
- Jupyter Notebook, JupyterLab or Google Colab

## **Installation**

```bash
git clone https://github.com/tasleemkhan-code/EnsembleLearning-Model.git
cd EnsembleLearning-Model
pip install scikit-learn xgboost seaborn pandas jupyter
```

## **Run**

```bash
jupyter notebook EnsembleLearning.ipynb
```

Then run all cells from top to bottom. The notebook can also be opened directly in Google Colab.

# **Project Structure**

```
EnsembleLearning-Model/
├── EnsembleLearning.ipynb   # Data loading, training and evaluation of all ensemble models
└── README.md
```

# **Notes**

- The base `LogisticRegression` inside the stacking ensemble uses the default `max_iter` and raises a convergence warning. Scaling the features or increasing `max_iter` resolves it.
- Newer XGBoost versions ignore the `use_label_encoder` parameter, so it can be removed from the `XGBClassifier` call.

# **Tech Stack**

Python, scikit-learn, XGBoost, Seaborn, Pandas, Jupyter Notebook

# **Author**

**Tasleem Khan**
GitHub: [@tasleemkhan-code](https://github.com/tasleemkhan-code)
LinkedIn: [tasleemkhan-9477b5306](https://linkedin.com/in/tasleemkhan-9477b5306)
