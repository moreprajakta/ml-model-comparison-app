import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


@st.cache_data
def load_dataset(name: str):
    if name == "Iris":
        data = load_iris(as_frame=True)
    elif name == "Wine":
        data = load_wine(as_frame=True)
    elif name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    df = data.frame.copy()
    target_name = data.target_names
    df["target"] = data.target
    return df, "target", list(data.feature_names)


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in encoded.columns:
        if encoded[col].dtype == object or encoded[col].dtype.name == "category":
            encoder = LabelEncoder()
            encoded[col] = encoder.fit_transform(encoded[col].astype(str))
    return encoded


def build_model(name: str, params: dict):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=params["max_iter"], C=params["C"], solver="liblinear")
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"] or None, random_state=42)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Report": classification_report(y_test, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Predictions": (y_test, y_pred),
    }


def show_confusion_matrix(matrix, classes):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="ML Model Comparison App", layout="wide")
    st.title("ML Model Comparison App")
    st.write(
        "Compare Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest on sample data or your own CSV dataset."
    )

    with st.sidebar:
        st.header("Dataset settings")
        dataset_choice = st.radio(
            "Choose dataset",
            ["Iris", "Wine", "Breast Cancer", "Custom CSV"],
            index=0,
        )

        custom_file = None
        if dataset_choice == "Custom CSV":
            custom_file = st.file_uploader("Upload a CSV file", type=["csv"])

        st.markdown("---")
        st.header("Train / test split")
        test_size = st.slider("Test set fraction", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        random_state = st.number_input("Random seed", value=42, step=1)
        scale_features = st.checkbox("Scale numeric features", value=True)

        st.markdown("---")
        st.header("Select models")
        selected_models = st.multiselect(
            "Pick models to compare",
            ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"],
            default=["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"],
        )

        st.markdown("---")
        st.header("Model hyperparameters")
        with st.expander("Logistic Regression"):
            log_reg_C = st.number_input("Inverse regularization strength (C)", min_value=0.01, max_value=10.0, value=1.0)
            log_reg_max_iter = st.slider("Max iterations", min_value=50, max_value=500, value=200, step=10)
        with st.expander("Decision Tree"):
            dt_max_depth = st.slider("Max depth", min_value=1, max_value=30, value=5)
        with st.expander("kNN"):
            knn_neighbors = st.slider("Number of neighbors", min_value=1, max_value=15, value=5)
        with st.expander("Random Forest"):
            rf_n_estimators = st.slider("Number of trees", min_value=10, max_value=200, value=100, step=10)
            rf_max_depth = st.slider("Max depth", min_value=1, max_value=30, value=7)

    if dataset_choice == "Custom CSV" and custom_file is None:
        st.warning("Upload a CSV file to train models on your own dataset.")
        st.stop()

    if dataset_choice == "Custom CSV":
        try:
            df = pd.read_csv(custom_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()

        if df.empty:
            st.error("The uploaded file is empty.")
            st.stop()

        st.subheader("Custom dataset preview")
        st.dataframe(df.head())

        default_target_index = df.columns.get_loc("target") if "target" in df.columns else 0
        target_column = st.selectbox(
            "Choose the target column",
            options=df.columns,
            index=default_target_index,
        )
        feature_columns = st.multiselect(
            "Choose feature columns",
            options=[col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column],
        )
    else:
        df, target_column, feature_columns = load_dataset(dataset_choice)
        st.subheader(f"{dataset_choice} dataset preview")
        st.dataframe(df.head())
        st.markdown(f"**Target column:** `{target_column}`")

    if not feature_columns:
        st.error("Please select at least one feature column.")
        st.stop()

    if target_column not in df.columns:
        st.error("Please select a valid target column.")
        st.stop()

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if y.isna().any():
        missing_target = y.isna().sum()
        st.warning(f"Target column has {missing_target} missing value(s). These rows will be removed.")
        keep_index = y.notna()
        X = X.loc[keep_index]
        y = y.loc[keep_index]

    if X.isna().any().any():
        missing_features = X.isna().sum().sum()
        st.warning(f"Feature data has {missing_features} missing value(s). Rows with missing values will be removed.")
        keep_index = X.dropna().index
        X = X.loc[keep_index]
        y = y.loc[keep_index]

    if X.empty or y.empty:
        st.error("No valid rows remain after removing missing data. Please upload a dataset with complete rows.")
        st.stop()

    X = encode_categorical(X)
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if y.dtype == object or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y.astype(str))

    if len(pd.unique(y)) < 2:
        st.error(
            "The chosen target column contains only one class. "
            "Please select a target with at least two distinct labels."
        )
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(pd.unique(y)) > 1 else None
    )

    if len(pd.unique(y_train)) < 2:
        st.error(
            "The training split contains only one class. "
            "Try reducing the test set size or using a dataset with more class variety."
        )
        st.stop()

    if not selected_models:
        st.error("Select at least one model to compare.")
        st.stop()

    results = []
    detailed = {}
    for model_name in selected_models:
        params = {
            "C": log_reg_C,
            "max_iter": log_reg_max_iter,
            "max_depth": dt_max_depth,
            "n_neighbors": knn_neighbors,
            "n_estimators": rf_n_estimators,
        }
        model = build_model(model_name, params)
        try:
            stats = evaluate_model(model, X_train, X_test, y_train, y_test)
        except Exception as e:
            st.error(f"Training failed for {model_name}: {e}")
            st.stop()

        results.append(
            {
                "Model": model_name,
                "Accuracy": stats["Accuracy"],
                "Precision": stats["Precision"],
                "Recall": stats["Recall"],
                "F1 Score": stats["F1 Score"],
            }
        )
        detailed[model_name] = stats

    st.subheader("Model comparison")
    results_df = pd.DataFrame(results).set_index("Model")
    st.dataframe(results_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1 Score": "{:.3f}"}))

    if len(selected_models) == 1:
        chosen = selected_models[0]
        stats = detailed[chosen]
        st.subheader(f"Detailed results for {chosen}")
        st.write("### Classification report")
        st.text(stats["Report"])
        st.write("### Confusion matrix")
        class_labels = [str(c) for c in sorted(pd.unique(y))]
        show_confusion_matrix(stats["Confusion Matrix"], class_labels)
    else:
        st.subheader("Detailed results")
        selected_detail = st.selectbox("Choose a model to inspect", selected_models)
        stats = detailed[selected_detail]
        st.write("### Classification report")
        st.text(stats["Report"])
        st.write("### Confusion matrix")
        class_labels = [str(c) for c in sorted(pd.unique(y))]
        show_confusion_matrix(stats["Confusion Matrix"], class_labels)

    st.sidebar.markdown("---")
    st.sidebar.write(
        "This Streamlit app compares the five classification models from the assignment notebook. "
        "Use a built-in dataset or upload your own CSV file."
    )


if __name__ == "__main__":
    main()
