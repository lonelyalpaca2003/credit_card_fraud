# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using anonymized transaction data from European cardholders.

## Problem Statement

Credit card fraud costs the financial industry billions annually and erodes customer trust. Detecting fraudulent transactions in real-time is critical, but challenging due to the rarity of fraud cases and the need to minimize false positives that inconvenience legitimate customers.

**Business Question:** Can we accurately identify fraudulent transactions while maintaining a low false positive rate to ensure genuine customers aren't disrupted?

## Dataset Overview

**Source:** Credit Card Fraud Detection Dataset 2023

**Target Variable:** Class (1 = fraudulent, 0 = legitimate)

**Features:**
- **Transaction Attributes:** V1-V28 (anonymized features representing time, location, and other transaction characteristics)
- **Financial:** Amount (transaction value)
- **Identifier:** id (unique transaction ID)

**Dataset Characteristics:**
- 550,000+ transactions
- Highly imbalanced (fraudulent transactions are rare)
- Anonymized for privacy compliance

## Methodology

**1. Exploratory Data Analysis**
- Class distribution analysis
- Transaction amount patterns
- Feature correlation exploration
- Identifying fraud indicators

**2. Data Preprocessing**
- Handling class imbalance
- Feature scaling
- Train-test split with stratification

**3. Evaluation Strategy**
- Precision, Recall, F1-Score (accuracy is misleading for imbalanced data)
- ROC-AUC and Precision-Recall curves
- Confusion matrix analysis
- Cost-sensitive evaluation

## Project Structure
