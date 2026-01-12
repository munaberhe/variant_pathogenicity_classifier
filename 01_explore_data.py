import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/clinvar_sample.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print("\nHead of the dataset:")
    print(df.head())

    print("\nClinical significance value counts:")
    print(df["clinical_significance"].value_counts())

    # Basic label binarisation for summary: Pathogenic vs Benign/Other
    df["is_pathogenic"] = df["clinical_significance"].isin(
        ["Pathogenic", "Likely_pathogenic"]
    )

    print("\nPathogenic vs non-pathogenic counts:")
    print(df["is_pathogenic"].value_counts())

    # Plot allele frequency distribution by label (if 'af' exists)
    if "af" in df.columns:
        plt.figure()
        df.boxplot(column="af", by="is_pathogenic")
        plt.suptitle("")  # remove automatic title
        plt.title("Allele frequency by pathogenic label")
        plt.xlabel("Is pathogenic")
        plt.ylabel("Allele frequency")
        plt.tight_layout()
        out_path = FIG_DIR / "af_by_label_boxplot.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"\nSaved boxplot to {out_path}")


if __name__ == "__main__":
    main()

