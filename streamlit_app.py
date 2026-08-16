import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = './heart.csv'


@st.cache_data
def load_heart_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Please add the heart disease CSV file in the repository."
        )
    df = pd.read_csv(path)
    if 'target' not in df.columns:
        raise ValueError("Expected the dataset to contain a 'target' column.")
    return df


def build_models() -> dict:
    return {
        'Logistic Regression': LogisticRegression(
            penalty='l1', solver='liblinear', random_state=42, max_iter=200
        ),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
    }


def load_saved_models(model_folder: str = 'model') -> dict:
    models = {}
    mapping = {
        'Logistic Regression': 'Logistic_Regression.joblib',
        'Decision Tree': 'Decision_Tree.joblib',
        'kNN': 'kNN.joblib',
        'Naive Bayes': 'Naive_Bayes.joblib',
        'Random Forest': 'Random_Forest.joblib',
    }
    for name, fname in mapping.items():
        path = os.path.join(model_folder, fname)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                # Ignore load errors; user will see training fallback
                pass
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, fit_model: bool = True):
    if fit_model:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    stats = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Report': classification_report(y_test, y_pred, zero_division=0),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
    }

    if y_proba is not None:
        stats['AUC'] = roc_auc_score(y_test, y_proba)
    else:
        stats['AUC'] = None

    return stats


def show_confusion_matrix(matrix, classes):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='black')
    st.pyplot(fig)


def main():
    st.set_page_config(page_title='Heart Disease Model Comparison', layout='wide')
    st.title('Heart Disease Classification App')
    st.write(
        'This app uses the heart disease dataset and the five classification models from the assignment notebook. '
        'It is for display only and does not support alternate datasets.'
    )

    try:
        df = load_heart_dataset()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader('Dataset preview')
    st.dataframe(df.head())
    st.markdown(f'**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns')
    st.markdown('**Target column:** `target`')

    st.sidebar.header('Options')
    uploaded_file = st.sidebar.file_uploader('Upload CSV test data (optional)', type=['csv'])
    use_saved = st.sidebar.checkbox('Use saved models from /model folder (if available)')

    X = df.drop('target', axis=1)
    y = df['target']

    if X.isna().any().any() or y.isna().any():
        st.warning('Missing values detected. Rows with missing values will be removed.')
        valid_index = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_index]
        y = y.loc[valid_index]

    if X.empty or y.empty:
        st.error('No valid data remains after removing missing values.')
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # If user uploaded a test CSV, use it as the test set (must contain 'target')
    if uploaded_file is not None:
        try:
            df_test = pd.read_csv(uploaded_file)
            if 'target' not in df_test.columns:
                st.error('Uploaded CSV must contain a `target` column.')
                st.stop()
            X_test = df_test.drop('target', axis=1)
            y_test = df_test['target']
            # align columns if necessary
            if list(X_test.columns) != list(X.columns):
                st.warning('Uploaded test CSV columns differ from training dataset — attempting to align by column names.')
                X_test = X_test.reindex(columns=X.columns)
            # scale using scaler fitted on training set
            X_test = scaler.transform(X_test.fillna(0))
            X_train = X_scaled
            y_train = y
        except Exception as exc:
            st.error(f'Failed to read uploaded CSV: {exc}')
            st.stop()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(pd.unique(y)) > 1 else None
        )

    models = build_models()
    # optionally load saved pre-trained models
    saved_models = load_saved_models() if use_saved else {}
    results = []
    detailed = {}

    for model_name, model in models.items():
        try:
            if model_name in saved_models:
                stats = evaluate_model(saved_models[model_name], X_train, X_test, y_train, y_test, fit_model=False)
            else:
                stats = evaluate_model(model, X_train, X_test, y_train, y_test, fit_model=True)
        except Exception as exc:
            st.error(f'Evaluation failed for {model_name}: {exc}')
            st.stop()

        results.append(
            {
                'Model': model_name,
                'Accuracy': stats['Accuracy'],
                'Precision': stats['Precision'],
                'Recall': stats['Recall'],
                'F1 Score': stats['F1 Score'],
                'AUC': stats['AUC'],
                'MCC': stats['MCC'],
            }
        )
        detailed[model_name] = stats

    results_df = pd.DataFrame(results).set_index('Model')
    st.subheader('Model comparison')
    st.dataframe(
        results_df.style.format(
            {
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1 Score': '{:.3f}',
                'AUC': '{:.3f}',
                'MCC': '{:.3f}',
            }
        )
    )

    selected_model = st.selectbox('View detailed model results', list(models.keys()))
    stats = detailed[selected_model]

    st.subheader(f'Detailed results for {selected_model}')
    st.write('### Classification report')
    st.text(stats['Report'])
    st.write('### Confusion matrix')
    class_labels = [str(c) for c in sorted(pd.unique(y))]
    show_confusion_matrix(stats['Confusion Matrix'], class_labels)

    st.sidebar.markdown('---')
    st.sidebar.write(
        'This app is aligned with the assignment notebook: it uses the heart disease dataset and the five specified classifiers only.'
    )


if __name__ == '__main__':
    main()
