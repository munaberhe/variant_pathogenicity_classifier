import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("data/clinvar_sample.csv")


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Define binary label: pathogenic vs non-pathogenic
    pathogenic_labels = {"Pathogenic", "Likely_pathogenic"}
    df = df[df["clinical_significance"].notna()].copy()
    df["is_pathogenic"] = df["clinical_significance"].isin(pathogenic_labels).astype(int)

    # Example features; adjust based on your CSV columns
    feature_cols = ["gene", "consequence", "impact", "polyphen", "af"]
    X = df[feature_cols]
    y = df["is_pathogenic"]

    # Identify categorical vs numeric features
    categorical_features = ["gene", "consequence", "impact"]
    numeric_features = ["polyphen", "af"]

    # Preprocess: one-hot encode categorical, scale numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    # Model: RandomForest
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nFitting model...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("\nClassification report (RandomForest on variant features):")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")


if __name__ == "__main__":
    main()

