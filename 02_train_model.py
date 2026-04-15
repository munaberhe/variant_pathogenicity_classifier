"""
Train a Random Forest classifier to predict variant pathogenicity
from real ClinVar data filtered to five cardiac disease genes.

Features used:
  gene            — which cardiac gene (MYBPC3, MYH7, SCN5A, KCNQ1, LMNA)
  variant_type    — molecular type (single nucleotide variant, Deletion, etc.)
  review_status   — ClinVar evidence strength (expert panel, criteria provided, etc.)
  n_submitters    — number of independent submitters (more = stronger evidence)
  origin          — germline vs somatic

Target:
  is_pathogenic = 1 if Pathogenic or Likely pathogenic, 0 if Benign or Likely benign

Run 00_download_filter.py first to generate data/clinvar_cardiac.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
DATA_PATH = Path("data/clinvar_cardiac.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic"}
CATEGORICAL = ["gene", "variant_type", "review_status", "origin"]
NUMERIC = ["n_submitters"]
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def plot_feature_importance(pipe: Pipeline, out_path: Path, top_n: int = 20) -> None:
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
    feature_names = cat_names + NUMERIC

    importances = pipe.named_steps["model"].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[idx[::-1]], color="steelblue", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
    ax.set_xlabel("Mean decrease in impurity (feature importance)")
    ax.set_title(f"Top {top_n} features — cardiac variant pathogenicity classifier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["clinical_significance"].notna()].copy()
    df["is_pathogenic"] = df["clinical_significance"].isin(PATHOGENIC_LABELS).astype(int)
    df = df.dropna(subset=CATEGORICAL + NUMERIC)

    X = df[CATEGORICAL + NUMERIC]
    y = df["is_pathogenic"]

    print(f"Dataset: {len(df):,} variants")
    print(f"  Pathogenic:     {y.sum():,}")
    print(f"  Non-pathogenic: {(y == 0).sum():,}")
    print(f"  Genes: {sorted(df['gene'].unique())}")

    pipe = build_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nFitting model...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Pathogenic"]))
    print(f"ROC-AUC (held-out test set): {roc_auc_score(y_test, y_proba):.3f}")

    # 5-fold cross-validation for a more reliable estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC:           {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    plot_feature_importance(pipe, FIG_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
