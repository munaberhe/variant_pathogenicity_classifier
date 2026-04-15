"""
SHAP interpretability analysis for the cardiac variant pathogenicity classifier.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution score
for every individual prediction. In clinical genomics this matters: if a model
flags a variant in MYBPC3 as pathogenic, you want to know whether that call was
driven by a high-confidence expert-panel review status, or by some artefact.

Run 00_download_filter.py then 02_train_model.py before this script.

Install: pip install shap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap

# ---------------------------------------------------------------------------
DATA_PATH = Path("data/clinvar_cardiac.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic"}
CATEGORICAL = ["gene", "variant_type", "review_status", "origin"]
NUMERIC = ["n_submitters"]
# ---------------------------------------------------------------------------


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df[df["clinical_significance"].notna()].copy()
    df["is_pathogenic"] = df["clinical_significance"].isin(PATHOGENIC_LABELS).astype(int)
    df = df.dropna(subset=CATEGORICAL + NUMERIC)

    X = df[CATEGORICAL + NUMERIC]
    y = df["is_pathogenic"]

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
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # Transform test set and recover feature names
    X_test_t = pipe.named_steps["preprocess"].transform(X_test)
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    feature_names = list(ohe.get_feature_names_out(CATEGORICAL)) + NUMERIC

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_test_t)

    # shap_values is a list [neg_class, pos_class] for binary RF
    sv_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

    # --- Plot 1: beeswarm summary (shows direction and magnitude) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_pos,
        X_test_t,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP feature contributions — pathogenic class", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # --- Plot 2: mean |SHAP| bar chart ---
    mean_abs = np.abs(sv_pos).mean(axis=0)
    idx = np.argsort(mean_abs)[-20:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), mean_abs[idx], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title("SHAP mean absolute impact — top 20 features", fontsize=13)
    plt.tight_layout()
    out = FIG_DIR / "shap_bar.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    print("\nDone. SHAP plots saved to figures/")


if __name__ == "__main__":
    main()
