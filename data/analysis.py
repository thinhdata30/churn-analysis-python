import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------

df = pd.read_csv("data/Churn.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna().copy()

# Convert target variable to binary
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# --------------------------------------------------
# 3. Exploratory Data Analysis
# --------------------------------------------------

churn_rate = df["Churn"].mean()

print(f"\nOverall Churn Rate: {churn_rate:.2%}")

print("\nChurn Rate by Contract:")
print(
    df.groupby("Contract")["Churn"]
    .mean()
    .sort_values(ascending=False)
)

print("\nChurn Rate by Internet Service:")
print(
    df.groupby("InternetService")["Churn"]
    .mean()
    .sort_values(ascending=False)
)


# --------------------------------------------------
# 4. Prepare Data for Machine Learning
# --------------------------------------------------

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 6. Train Logistic Regression Model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Model Evaluation
# --------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"\nModel Accuracy: {accuracy:.3f}")
print(f"ROC-AUC Score: {roc_auc:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 8. Create Churn Visualization
# --------------------------------------------------

churn_by_contract = (
    df.groupby("Contract")["Churn"]
    .mean()
    .sort_values(ascending=False)
)

churn_by_contract.plot(
    kind="bar"
)

plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "churn_by_contract.png",
    dpi=300
)

plt.close()

print("\nAnalysis complete.")
print("Visualization saved as churn_by_contract.png")
