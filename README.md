# ml-model-comparison-app
 Implements Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest on a chosen dataset. Includes evaluation metrics, comparison tables, and an interactive Streamlit app deployed on Streamlit Cloud for model selection and performance visualization.

## Problem statement
Build and compare multiple classification models on a public dataset, report evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), provide observations and deploy an interactive Streamlit app.

## Dataset description
- Dataset file: `heart.csv`
- Instances: 1025
- Features (excluding `target`): 13
- Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
- Source: included in repository (common heart disease dataset)

## Github Repository Link
https://github.com/moreprajakta/ml-model-comparison-app/tree/main

## Models used and Evaluation Metrics
The project implements the following models:
- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)

### Evaluation Metrics Comparison Table

**Training Dataset:** heart.csv (1025 instances, 13 features)  
**Test Dataset:** test_data.csv (60 instances)  
**Preprocessing:** StandardScaler normalization applied to all features

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.820 | 0.935 | 0.770 | 0.924 | 0.840 | 0.651 |
| Decision Tree | 0.985 | 0.986 | 1.000 | 0.971 | 0.986 | 0.971 |
| kNN | 0.698 | 0.834 | 0.713 | 0.686 | 0.699 | 0.396 |
| Naive Bayes | 0.829 | 0.904 | 0.807 | 0.876 | 0.840 | 0.660 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Table Description:**
- **Accuracy:** Percentage of correct predictions out of total predictions
- **AUC (Area Under Curve):** Measures model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)
- **Precision:** Of positive predictions, how many were correct (TP / (TP + FP))
- **Recall:** Of actual positive cases, how many were correctly identified (TP / (TP + FN))
- **F1 Score:** Harmonic mean of Precision and Recall (2 × (Precision × Recall) / (Precision + Recall))
- **MCC:** Matthews Correlation Coefficient, a robust metric for class imbalance (-1 = completely wrong, 0 = random, 1 = perfect)

## Model observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Provides a solid baseline with strong AUC score (0.935) and excellent recall (0.924), making it reliable for identifying positive cases. However, lower precision (0.770) suggests some false positives. Good interpretability and computational efficiency. |
| **Decision Tree** | Achieves very high accuracy (0.985) and perfect precision/recall on test set. Decision boundaries are interpretable and easy to visualize. May indicate potential overfitting—requires cross-validation to confirm generalization capability. |
| **kNN** | Shows moderate performance (Accuracy 0.698, F1 0.699) with lowest MCC score (0.396), suggesting struggles with class separation. Sensitive to feature scaling and data imbalance. Computationally expensive for large datasets but can work well with proper hyperparameter tuning. |
| **Naive Bayes** | Good performance with balanced metrics (Accuracy 0.829, F1 0.840) and strong AUC (0.904). Performs well as a fast baseline model with reasonable recall (0.876). Assumes feature independence which may not hold perfectly for this dataset. |
| **Random Forest** | Achieves perfect scores across all metrics (Accuracy 1.000, AUC 1.000, F1 1.000). Excellent ensemble method that captures complex patterns. Perfect scores warrant validation with cross-validation and additional holdout test sets to rule out overfitting or test/train data leakage. |

## Overall winner
**Random Forest** is the best-performing model on this dataset, achieving perfect scores (1.0) across all evaluation metrics. However, the perfect scores suggest possible overfitting; therefore, cross-validation and testing on completely unseen data are recommended to confirm true generalization performance. Among models with more conservative scores, **Logistic Regression** and **Naive Bayes** offer reliable alternatives with good AUC and recall values for practical deployment.
