"""
Download and filter real ClinVar variant data to cardiac disease genes.

I'm a cardiac physiologist — these five genes cover the conditions I see most
in the echo lab and clinical genetics setting:

  MYBPC3  — most common cause of HCM; truncating variants dominate
  MYH7    — HCM and DCM; missense variants in the myosin head region
  SCN5A   — Brugada syndrome, LQT3, progressive cardiac conduction disease
  KCNQ1   — LQT1; the most common inherited long QT gene
  LMNA    — DCM with conduction disease; high risk of sudden death

Data source: ClinVar variant_summary.txt (NCBI FTP, GRCh38 assembly).
Filters: cardiac genes only, clear-cut Pathogenic/Likely pathogenic/Benign/Likely benign labels.
VUS variants are excluded here — in practice they are the hardest clinical cases.
"""

import gzip
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
CARDIAC_GENES = {"MYBPC3", "MYH7", "SCN5A", "KCNQ1", "LMNA"}

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

DATA_DIR = Path("data")
GZ_PATH = DATA_DIR / "variant_summary.txt.gz"
OUT_PATH = DATA_DIR / "clinvar_cardiac.csv"

KEEP_COLS = [
    "GeneSymbol",
    "Type",
    "ClinicalSignificance",
    "ReviewStatus",
    "NumberSubmitters",
    "Origin",
    "Assembly",
]

RENAME = {
    "GeneSymbol": "gene",
    "Type": "variant_type",
    "ClinicalSignificance": "clinical_significance",
    "ReviewStatus": "review_status",
    "NumberSubmitters": "n_submitters",
    "Origin": "origin",
}

KEEP_SIGS = {
    "Pathogenic",
    "Likely pathogenic",
    "Benign",
    "Likely benign",
}
# ---------------------------------------------------------------------------


def download(url: str, dest: Path) -> None:
    print(f"Downloading ClinVar variant summary (~200 MB)...")
    print(f"  {url}")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.\n")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    if not GZ_PATH.exists():
        download(CLINVAR_URL, GZ_PATH)
    else:
        print(f"Using cached file: {GZ_PATH}\n")

    print("Streaming and filtering ClinVar data (GRCh38, cardiac genes)...")
    chunks = []
    with gzip.open(GZ_PATH, "rt", encoding="utf-8") as fh:
        for chunk in pd.read_csv(fh, sep="\t", chunksize=50_000, low_memory=False):
            # Restrict to GRCh38
            if "Assembly" in chunk.columns:
                chunk = chunk[chunk["Assembly"] == "GRCh38"]
            # Filter to our cardiac genes
            sub = chunk[chunk["GeneSymbol"].isin(CARDIAC_GENES)]
            if len(sub):
                chunks.append(sub[[c for c in KEEP_COLS if c in sub.columns]])

    if not chunks:
        raise RuntimeError("No matching variants found — check column names in the download.")

    df = pd.concat(chunks, ignore_index=True)
    df.rename(columns=RENAME, inplace=True)
    df.drop(columns=["Assembly"], inplace=True, errors="ignore")

    # Keep only unambiguous pathogenic / benign labels
    df = df[df["clinical_significance"].isin(KEEP_SIGS)].copy()

    print(f"\nVariants retained after filtering: {len(df):,}")
    print("\nVariants per gene:")
    print(df["gene"].value_counts().to_string())
    print("\nClinical significance breakdown:")
    print(df["clinical_significance"].value_counts().to_string())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
    print("Ready — run 01_explore_data.py next.")


if __name__ == "__main__":
    main()
