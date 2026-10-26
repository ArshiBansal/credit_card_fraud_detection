<div align="center">

# Credit Card Fraud Detection

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=36BCF7&center=true&vCenter=true&width=720&lines=One+transaction.+A+signal+in+the+noise.;Find+the+fraud.+Understand+the+why.;Turn+scores+into+cost-aware+decisions." alt="Animated text: Find the fraud, understand the why, and make cost-aware decisions">

**A transaction is only a row of numbers until something looks different.** This project follows that difference from first exploration to an explainable, cost-aware fraud decision.

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10 or newer">
<img src="https://img.shields.io/badge/Jupyter-Notebook%20workflow-F37626" alt="Jupyter notebook workflow">
<img src="https://img.shields.io/badge/Focus-Imbalanced%20learning-8A2BE2" alt="Imbalanced learning">
<img src="https://img.shields.io/badge/Explainability-SHAP-00A98F" alt="SHAP explainability">

</div>

> [!NOTE]
> This is a learning and analysis project built around anonymized, PCA-transformed transaction data. It is **not** a production fraud-screening service.

## The story behind the project

Imagine scanning a stream of card transactions and looking for a handful of fraudulent ones. Most transactions are legitimate. The rare cases we care about are surrounded by a sea of ordinary activity, so a model that simply says “legitimate” most of the time can appear accurate while missing the point entirely.

That makes fraud detection a series of questions, not just a model-training exercise:

1. **What does the data look like?** Explore transaction features and prepare a stratified split without losing the rare fraud examples.
2. **Can a model find the signal?** Compare classifiers and imbalance-aware approaches using metrics that reveal how well fraud is detected.
3. **Why did it raise an alert?** Use SHAP to examine which features influence predictions, globally and for individual transactions.
4. **When should an alert be raised?** Choose a threshold with explicit costs in mind, balancing missed fraud against false alarms.

The result is a practical walkthrough from raw data to a decision—and a reminder that a prediction score is only useful when its consequences are understood.

<details>
<summary><strong>Project snapshot</strong></summary>

| Item | Details |
| --- | --- |
| Prediction | Fraud (`1`) or non-fraud (`0`) |
| Processed features | 30 columns, including PCA components |
| Data split | 227,845 training and 56,962 test transactions |
| Analysis | Four notebooks: preparation, modeling, explainability, and cost evaluation |
| Saved outputs | Parquet data, serialized model, comparison metrics, threshold settings, and SHAP artifacts |

</details>

## Follow the investigation

Run the notebooks in order. Each one picks up where the last left off.

| Chapter | Notebook | In this chapter |
| ---: | --- | --- |
| 1 — The evidence | [`01_EDA_and_Preprocessing.ipynb`](notebooks/01_EDA_and_Preprocessing.ipynb) | Explore distributions, scale `Time` and `Amount`, and create a stratified train/test split. |
| 2 — The search | [`02_Imbalance_Handling_and_Modeling.ipynb`](notebooks/02_Imbalance_Handling_and_Modeling.ipynb) | Compare Logistic Regression, Random Forest, XGBoost, and LightGBM with imbalance-handling approaches. |
| 3 — The explanation | [`03_Explainability_SHAP.ipynb`](notebooks/03_Explainability_SHAP.ipynb) | Examine global feature importance and explanations for individual predictions. |
| 4 — The decision | [`04_Cost_Sensitive_Evaluation.ipynb`](notebooks/04_Cost_Sensitive_Evaluation.ipynb) | Compare thresholds using configurable false-positive, false-negative, and true-positive costs. |

> **The path:** evidence → model → explanation → decision

## Start the investigation

### What you need

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)
- The `datasets` package if you want to download the source data and rerun Notebook 01

### Set up the environment

From the repository root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install datasets
```

> [!TIP]
> Processed data and saved model outputs are included, so you can explore notebooks 02–04 without rerunning Notebook 01. Install `datasets` when you want to fetch and preprocess the source data again.

### Open the notebooks

```powershell
jupyter lab
```

Open the notebooks from `notebooks/` and run them there so their relative `../data/processed/` and `../models/` paths resolve as expected.

## What is in the repository?

```text
.
├── data/
│   └── processed/      # Train/test features and labels (Parquet)
├── models/
│   ├── shap/           # SHAP values, sample data, and feature importance
│   ├── best_fraud_model.pkl
│   ├── model_comparison_results.csv
│   └── optimal_threshold.json
├── notebooks/          # The four chapters of the analysis
└── requirements.txt
```

## A look at the saved results

In the checked-in model comparison, the XGBoost (`scale_pos_weight`) entry reports:

| Metric | Score |
| --- | ---: |
| Precision | `0.872` |
| Recall | `0.837` |
| F1 | `0.854` |
| PR-AUC | `0.879` |
| ROC-AUC | `0.979` |

The saved threshold configuration records a minimum-cost threshold of **`0.64`**, using a false-negative cost of `150`, a false-positive cost of `5`, and a true-positive cost of `10`. The recorded minimum test-set cost is `9160`.

<p>
<mark>These figures describe the checked-in experiment; they are not a promise of future performance.</mark>
Results depend on the data, model, and assumptions used.
</p>

## The decision is part of the model

A high ranking score does not decide how many transactions to flag. That depends on the threshold—and on what a missed fraud or a false alarm costs.

- **Look beyond accuracy.** With severe class imbalance, PR-AUC, precision, and recall are especially useful for understanding fraud detection performance.
- **Treat costs as assumptions.** The example cost values are not universal. Replace them with estimates grounded in the intended business context.
- **Keep evaluation honest.** The cost-sensitive notebook selects and reports thresholds against test labels. For an unbiased final evaluation, choose the threshold using training/validation data and reserve an untouched test set for final measurement.
- **Question defaults.** A `0.50` threshold is ~~always the right answer~~ simply a starting point; the operating point should reflect the real trade-off between missed fraud and false alerts.

> A score estimates risk. A threshold turns that estimate into an action. The cost assumptions explain why that action might make sense.

> [!WARNING]
> This project does not include live transaction ingestion, deployment, monitoring, or operational controls. Do not use the saved model or threshold for real-world decisions without suitable validation, oversight, and domain review.

## Where the data comes from

Notebook 01 loads [`Barvero/Credit_Card_Fraud_Analysis_Project`](https://huggingface.co/datasets/Barvero/Credit_Card_Fraud_Analysis_Project) from Hugging Face. The dataset is based on the classic anonymized credit card fraud dataset; PCA-transformed features are provided alongside `Time` and `Amount` for preprocessing.

---

<div align="center">

**Follow the signal. Understand the alert. Make the trade-off explicit.**

</div>
