"""
Exploratory analysis of ClinVar cardiac variant data.

Run 00_download_filter.py first to generate data/clinvar_cardiac.csv.

Genes covered:
  MYBPC3  — HCM (most common cause)
  MYH7    — HCM / DCM
  SCN5A   — Brugada syndrome, LQT3
  KCNQ1   — LQT1
  LMNA    — DCM with conduction disease
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/clinvar_cardiac.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nClinical significance counts:")
    print(df["clinical_significance"].value_counts().to_string())

    print("\nVariants per gene:")
    print(df["gene"].value_counts().to_string())

    print("\nVariant types observed:")
    print(df["variant_type"].value_counts().to_string())

    print("\nReview status breakdown:")
    print(df["review_status"].value_counts().to_string())

    # Binary label
    df["is_pathogenic"] = df["clinical_significance"].isin(
        ["Pathogenic", "Likely pathogenic"]
    ).astype(int)

    print(f"\nPathogenic:     {df['is_pathogenic'].sum():,}")
    print(f"Non-pathogenic: {(df['is_pathogenic'] == 0).sum():,}")

    # --- Plot 1: variant type by gene ---
    ct = (
        df.groupby(["gene", "variant_type"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ct.plot(kind="bar", ax=ax)
    ax.set_title("Variant type distribution across cardiac disease genes", fontsize=13)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Count")
    ax.legend(title="Variant type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    out = FIG_DIR / "variant_type_by_gene.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")

    # --- Plot 2: pathogenic fraction by gene ---
    frac = df.groupby("gene")["is_pathogenic"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    frac.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Proportion of pathogenic variants per cardiac gene", fontsize=13)
    ax.set_ylabel("Fraction pathogenic")
    ax.set_xlabel("Gene")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    out = FIG_DIR / "pathogenic_fraction_by_gene.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # --- Plot 3: review status distribution ---
    fig, ax = plt.subplots(figsize=(9, 4))
    df["review_status"].value_counts().plot(kind="barh", ax=ax, color="coral", edgecolor="white")
    ax.set_title("ClinVar evidence strength (review status)", fontsize=13)
    ax.set_xlabel("Number of variants")
    plt.tight_layout()
    out = FIG_DIR / "review_status_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # --- Plot 4: number of submitters distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    df["n_submitters"].clip(upper=20).hist(bins=20, ax=ax, color="mediumseagreen", edgecolor="white")
    ax.set_title("Distribution of ClinVar submitter counts (capped at 20)", fontsize=13)
    ax.set_xlabel("Number of submitters")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = FIG_DIR / "n_submitters_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
