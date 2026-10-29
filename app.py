"""Interactive customer churn demo using the existing Telco Churn dataset.

Run from the root of the repository:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "Churn.csv"


@st.cache_resource(show_spinner="Training the churn prediction model...")
def load_and_train_model():
    """Reproduce the existing train/test setup and return evaluation metrics."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].astype(int)

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, scores),
        "recall": recall_score(y_test, predicted),
        "accuracy": accuracy_score(y_test, predicted),
        "report": pd.DataFrame(classification_report(y_test, predicted, output_dict=True)).T,
    }
    return model, X, metrics


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
    st.title("📊 Customer Churn Prediction")
    st.write(
        "An interactive machine learning demo built with Python, Scikit-learn, "
        "Logistic Regression, and Streamlit."
    )

    if not DATA_PATH.exists():
        st.error("Cannot find data/Churn.csv. Make sure app.py is in the repository root.")
        st.stop()

    try:
        model, X, metrics = load_and_train_model()
    except Exception as error:
        st.error(f"Could not train the model: {error}")
        st.stop()

    st.subheader("Model performance on held-out test data")
    metric_cols = st.columns(3)
    metric_cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    metric_cols[1].metric("Churn Recall", f"{metrics['recall']:.1%}")
    metric_cols[2].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.caption("80/20 stratified train/test split · random_state=42 · balanced Logistic Regression")

    st.subheader("Try a customer prediction")
    st.write("Change the customer details below and click **Predict churn**.")
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    inputs = {}
    with st.form("prediction_form"):
        left, right = st.columns(2)
        for index, column in enumerate(X.columns):
            with (left if index % 2 == 0 else right):
                if column in numeric_features:
                    observed_min = X[column].min()
                    observed_max = X[column].max()
                    observed_median = X[column].median()
                    if pd.api.types.is_integer_dtype(X[column]):
                        inputs[column] = st.number_input(
                            column,
                            min_value=int(observed_min),
                            max_value=int(observed_max),
                            value=int(observed_median),
                            step=1,
                        )
                    else:
                        inputs[column] = st.number_input(
                            column,
                            min_value=float(observed_min),
                            max_value=float(observed_max),
                            value=float(observed_median),
                            step=1.0,
                        )
                else:
                    options = sorted(X[column].dropna().astype(str).unique().tolist())
                    most_common = str(X[column].mode(dropna=True).iloc[0])
                    inputs[column] = st.selectbox(
                        column, options, index=options.index(most_common)
                    )
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        customer = pd.DataFrame([inputs], columns=X.columns)
        score = float(model.predict_proba(customer)[0, 1])
        predicted_label = int(model.predict(customer)[0])
        st.subheader("Prediction result")
        st.metric("Model churn score", f"{score:.1%}")
        if predicted_label == 1:
            st.warning("The model flags this customer as more likely to churn.")
        else:
            st.success("The model does not flag this customer as likely to churn.")
        st.caption(
            "This score is not a calibrated real-world probability. "
            "The model uses class_weight='balanced' and is for portfolio demonstration only."
        )

    with st.expander("See the detailed model evaluation"):
        st.dataframe(metrics["report"].round(3), use_container_width=True)
        st.write(
            "Preprocessing (imputation, scaling, one-hot encoding) is fitted on training "
            "data through a Scikit-learn Pipeline to prevent test-set leakage."
        )

    st.divider()
    st.caption("Dataset: Telco Customer Churn | Independent portfolio project")


if __name__ == "__main__":
    main()
