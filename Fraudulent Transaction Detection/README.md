# Fraud Detection Using Machine Learning

This project focuses on building a proactive fraud detection system using a transactional dataset containing over 6.3 million financial records. The dataset simulates real-world banking operations with a heavy class imbalance (only 0.13% fraudulent transactions).

The objective is to build a machine learning pipeline that identifies fraudulent transactions based on behavioral features and transaction patterns.

## Dataset

The dataset used in this project contains **6362620** transactions with the following key features:

- Transaction type (`TRANSFER`, `CASH_OUT`, etc.)
- Amount
- Origin and destination balances (before and after transaction)
- Fraud indicator (`isFraud`)

**Download Dataset:**  
[Click here to download Fraud Detection CSV](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)

## Features

- Data Cleaning and Preprocessing
- Feature Engineering:
  - Balance changes
  - Percentage of balance moved
  - Transaction timing (hour-based)
  - Error flag on mismatch between balances and amount
- Imbalanced data handling using **SMOTE**
- Model building with **Random Forest Classifier**
- Custom fraud detection pipeline with real-time prediction capabilities
- Evaluation using precision, recall, F1-score, and ROC-AUC
- Interpretability through feature importance

## Model Performance

| Metric         | Score        |
|----------------|--------------|
| Accuracy       | ~99.98%      |
| Precision (Fraud) | 94%       |
| Recall (Fraud) | 100%         |
| F1 Score (Fraud) | 97%        |
| ROC AUC Score  | 0.998        |

## Testing the Model

The model was tested on:

- **Known fraud cases**: All correctly detected.
- **Legit transactions**: No false positives.
- **Unseen transaction types (`PAYMENT`, `DEBIT`, `CASH_IN`)**: Handled without errors, correctly predicted as non-fraud.

## How to Run

1. Clone the repository or open the provided Jupyter Notebook.
2. Download the dataset using the link above and place it in your working directory.
3. Run the notebook cells step by step.
4. Use the prediction pipeline to test new transactions.

## Notes

- Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions.
- The model is trained exclusively on these two types.
- Other types are excluded from training but safely handled in the pipeline.
