import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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


def evaluate_model(model, X_train, X_test, y_train, y_test):
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

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(pd.unique(y)) > 1 else None
    )

    models = build_models()
    results = []
    detailed = {}

    for model_name, model in models.items():
        try:
            stats = evaluate_model(model, X_train, X_test, y_train, y_test)
        except Exception as exc:
            st.error(f'Training failed for {model_name}: {exc}')
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
