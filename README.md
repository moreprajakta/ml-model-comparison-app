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
Add your GitHub repository link here (required for submission).

## Models used and Evaluation Metrics
The project implements the following models:
- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)

### Comparison Table (fill with actual numbers after running the app)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.820 | 0.935 | 0.770 | 0.924 | 0.840 | 0.651 |
| Decision Tree | 0.985 | 0.986 | 1.000 | 0.971 | 0.986 | 0.971 |
| kNN | 0.698 | 0.834 | 0.713 | 0.686 | 0.699 | 0.396 |
| Naive Bayes | 0.829 | 0.904 | 0.807 | 0.876 | 0.840 | 0.660 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Model observations
Add your short observations about each model's performance here. Example:
- Logistic Regression: ...
- Decision Tree: ...

Suggested observations (edit as needed):
- **Logistic Regression:** Good baseline with strong AUC (0.935) and high recall (0.924). Performs well but slightly less precise than tree/ensemble.
- **Decision Tree:** Very high scores on test set (Accuracy 0.985). May be overfitting to training data—check cross-validation.
- **kNN:** Moderate performance (Accuracy 0.698); sensitive to feature scaling and dataset imbalance.
- **Naive Bayes:** Strong recall and balanced F1 (0.840); good simple baseline for this dataset.
- **Random Forest:** Perfect scores on provided test set—likely indicates overfitting or test/ train overlap; validate with cross-validation and unseen data.

## Overall winner
Based on the metrics above, `Random Forest` achieves the highest scores on the provided test set. However, several models (Decision Tree, kNN) also show perfect or near-perfect scores, suggesting a need to verify test independence and perform cross-validation before declaring a final winner.

## Overall winner
State which model performed best on the chosen dataset and why.

## Running locally
1. Install dependencies: `pip install -r requirements.txt`
2. Start the app: `streamlit run streamlit_app.py`

## Streamlit Cloud deployment
1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set the repository and branch to this project and use `streamlit_app.py` as the main file.
4. Once deployed, paste the live app URL here.

## Test data and BITS Virtual Lab screenshot
- Include `test_data.csv` in the repo for evaluation.
- Add a screenshot of assignment execution on BITS Virtual Lab in the repository and in your final PDF submission.

## Notes
- The app now supports uploading a CSV test file (must include `target`) via the sidebar. If a test CSV is uploaded the models are trained on the bundled `heart.csv` dataset and evaluated on the uploaded test set.
- The app supports loading saved `*.joblib` models from the `model/` folder when the sidebar option is checked.

## Live Streamlit App
The Streamlit application is ready for deployment on Streamlit Community Cloud. Once the repository is pushed to GitHub, the app can be launched from `streamlit_app.py`.

### Running locally
1. Install dependencies: `pip install -r requirements.txt`
2. Start the app: `streamlit run streamlit_app.py`

### Streamlit Cloud deployment
1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set the repository and branch to this project and use `streamlit_app.py` as the main file.
4. Once deployed, the app will open an interactive frontend for all supported ML models.

A standard share link looks like:

`https://share.streamlit.io/moreprajakta/ml-model-comparison-app/main/streamlit_app.py`

Replace `moreprajakta` with your GitHub username if you deploy from a fork or personal repository.
