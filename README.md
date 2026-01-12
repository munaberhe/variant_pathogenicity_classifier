# Variant Pathogenicity Classifier (ClinVar-Style Toy Project)

## Overview

This mini-project builds a simple variant pathogenicity classifier using ClinVar-style variant annotations. The goal is to predict whether a genetic variant is pathogenic or benign/other using basic features such as gene, consequence, predicted impact, PolyPhen-like score and allele frequency.

The project is designed as a small, self-contained example to support an MSc thesis or portfolio on genomic variant interpretation and to demonstrate practical skills in:

- working with variant-level tabular data
- exploratory data analysis in Python
- building and evaluating machine learning classifiers (RandomForest)
- documenting a reproducible analysis pipeline

## Dataset

We use a small, synthetic ClinVar-style CSV file:

data/clinvar_sample.csv

with columns such as:

- variant_id – variant identifier (e.g. rsID)
- gene – gene symbol (e.g. BRCA1, CFTR, TP53)
- consequence – functional consequence (e.g. missense_variant, synonymous_variant, stop_gained, intron_variant)
- impact – coarse impact category (e.g. HIGH, MODERATE, LOW, MODIFIER)
- polyphen – PolyPhen-like score (0–1, where higher ≈ more damaging)
- af – allele frequency in a reference population (e.g. gnomAD-like)
- clinical_significance – label similar to ClinVar (e.g. Pathogenic, Likely_pathogenic, Benign, Likely_benign, VUS)

For this project we define a binary label:

- Positive (pathogenic): Pathogenic, Likely_pathogenic
- Negative (non-pathogenic): all other labels (Benign, Likely_benign, VUS, etc.)

The provided CSV contains a small number of variants across a few genes (BRCA1, BRCA2, CFTR, LDLR, TP53 and some generic genes) to keep the example compact and easy to understand.

## Methods

### 1. Exploratory data analysis

Script: 01_explore_data.py

This script:

1. Loads data/clinvar_sample.csv into a pandas DataFrame.
2. Prints the first few rows to inspect the structure.
3. Prints value counts for clinical_significance to see class distribution.
4. Creates a binary label is_pathogenic:
   - True if clinical_significance is Pathogenic or Likely_pathogenic
   - False otherwise
5. Prints counts of is_pathogenic (how many pathogenic vs non-pathogenic variants).
6. If the column af is present, creates a boxplot of allele frequency by is_pathogenic and saves it as:
   - figures/af_by_label_boxplot.png

This gives a quick overview of class balance and the relationship between allele frequency and pathogenicity in the toy data.

### 2. Variant pathogenicity classification

Script: 02_train_model.py

This script trains and evaluates a RandomForest classifier.

Steps:

1. Load data/clinvar_sample.csv.
2. Drop rows with missing clinical_significance.
3. Create a binary label is_pathogenic:
   - 1 if clinical_significance is Pathogenic or Likely_pathogenic
   - 0 otherwise
4. Select a small set of features:
   - gene
   - consequence
   - impact
   - polyphen
   - af
5. Define preprocessing using sklearn ColumnTransformer:
   - Categorical features (gene, consequence, impact) are one-hot encoded, with unknown categories ignored at test time.
   - Numeric features (polyphen, af) are standardised using StandardScaler.
6. Build a scikit-learn Pipeline that chains:
   - preprocess: the ColumnTransformer
   - model: RandomForestClassifier with:
     - n_estimators = 300
     - random_state = 42
     - class_weight = "balanced" to help with class imbalance
7. Split the dataset into training and test sets:
   - 70% training, 30% test
   - stratified by the label to preserve class proportions
8. Fit the pipeline on the training data.
9. Evaluate on the test data:
   - print a full classification report (precision, recall, f1-score, support for each class)
   - compute ROC-AUC using predicted probabilities for the positive class.

This produces baseline performance metrics for a simple model on a small variant dataset.

## Setup

1. Create and activate a Python virtual environment (recommended):

   - python -m venv .venv
   - source .venv/bin/activate    (Windows PowerShell: .venv\Scripts\Activate.ps1)

2. Install dependencies:

   - pip install pandas scikit-learn matplotlib

3. Create the project structure and place the dataset:

   - Make sure the folders data and figures exist.
   - Save the CSV file as:
     - data/clinvar_sample.csv

The provided example CSV already follows the expected format.

## How to Run

From the project folder variant_pathogenicity_classifier:

1. Exploratory analysis

   - Run:
     - python 01_explore_data.py

   This will:
   - load data/clinvar_sample.csv
   - print the head of the dataset
   - print counts per clinical_significance category
   - create and summarise the binary is_pathogenic label
   - if af is present, save a boxplot to:
     - figures/af_by_label_boxplot.png

2. Train and evaluate the classifier

   - Run:
     - python 02_train_model.py

   This will:
   - load data/clinvar_sample.csv
   - build the is_pathogenic label
   - preprocess categorical and numeric features
   - train a RandomForest classifier
   - print a classification report to the terminal
   - compute and print the ROC-AUC

## Results

On the provided toy ClinVar-style dataset (clinvar_sample.csv), the data is deliberately small and clean, with a strong relationship between low allele frequency, high-impact consequences and pathogenic labels. With this setup, the RandomForest baseline achieves:

- Accuracy: 1.00
- Macro F1-score: 1.00
- ROC-AUC: 1.00

This perfect performance reflects the simplicity and small size of the synthetic dataset rather than a realistic diagnostic scenario. In practice, variant interpretation is far more challenging, with noisier labels, more ambiguous classes such as VUS, and many more features involved.

The key point of this project is to demonstrate:

- a complete pipeline from raw tabular variant annotations to an evaluated classifier
- how basic genomic features (gene, consequence, impact, PolyPhen-like score, allele frequency) can be used for supervised learning
- how to structure and document such a project for reproducibility and portfolio use

## Discussion

This project provides a compact example of supervised learning for variant interpretation, which is central to many tasks in clinical and research genomics, including:

- clinical variant triage and reporting in diagnostic labs
- prioritisation of variants in rare disease and cancer pipelines
- annotation and scoring of variants in pharmacogenomics and drug development

The current implementation is intentionally simple:

- a single RandomForest model
- a small set of intuitive features
- a small synthetic dataset

Despite this, it demonstrates important practical skills:

- loading and inspecting variant-level tabular data in pandas
- engineering and encoding categorical and numeric features using scikit-learn
- training and evaluating a supervised classifier with appropriate metrics (precision, recall, F1, ROC-AUC)
- packaging everything in a clear, reproducible structure with a README

Possible extensions include:

- adding more predictive features (e.g. CADD, SIFT, conservation scores, splice prediction scores, ClinVar review status)
- comparing multiple models (logistic regression, gradient boosting, XGBoost, etc.)
- moving from binary to multi-class classification (Pathogenic, Benign, VUS)
- more advanced treatment of class imbalance (e.g. SMOTE, cost-sensitive learning)
- more robust validation (cross-validation, independent test sets)

Together with other projects (scRNA label transfer, phenotype–disease matching with LLMs, RNA-seq differential expression and pathway analysis), this classifier showcases experience with variant-level modelling and skills that are highly relevant to bioinformatics roles in genomics and the pharmaceutical industry.

