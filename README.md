# Cardiac Variant Pathogenicity Classifier

Random Forest classifier trained on real ClinVar data to predict pathogenicity for variants in five cardiac disease genes: MYBPC3, MYH7, SCN5A, KCNQ1, LMNA.

I picked these genes because they're the ones I actually encounter clinically as a cardiac physiologist — HCM, DCM, channelopathies, LMNA-related conduction disease. Wanted to build something on data I understand rather than a generic example dataset.

Features used: gene, variant type, ClinVar review status, number of submitters, origin. Everything available directly from ClinVar — no extra annotation tools needed to reproduce. Adding gnomAD AF and in-silico scores (CADD, REVEL) would be the obvious next step.

VUS variants are excluded from training. They're genuinely ambiguous and would just add noise to a binary classifier.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib shap
```

## Usage

```bash
# 1. Download and filter real ClinVar data (~200 MB, one-time)
python 00_download_filter.py

# 2. Explore the data
python 01_explore_data.py

# 3. Train and evaluate
python 02_train_model.py

# 4. SHAP interpretability plots
python 03_shap_analysis.py
```

Figures saved to `figures/`. The SHAP beeswarm (`shap_beeswarm.png`) shows that review status and variant type carry most of the signal — which lines up with how ACMG classification actually works.

---

Author: Muna Berhe
