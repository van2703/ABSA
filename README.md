# Shopee Aspect-Based Sentiment Analysis using Traditional Machine Learning

This project focuses on Aspect-Based Sentiment Analysis (ABSA) for Vietnamese Shopee product reviews.

## Task

Given a customer review comment, the system predicts sentiment labels for four aspects:

1. Product quality (`chat_luong`)
2. Packaging (`dong_goi`)
3. Delivery (`van_chuyen`)
4. Price (`gia_ca`)

Each aspect has three possible sentiment labels:

- positive
- negative
- neutral

## Main Model

The main model is:

TF-IDF + Logistic Regression

Four separate Logistic Regression classifiers are trained, one for each aspect.

## Comparison Models

The main model will be compared with:

- TF-IDF + Multinomial Naive Bayes
- TF-IDF + Linear SVM
- TF-IDF + Random Forest

## Project Structure

```text
data/
notebooks/
src/
results/
models/
report/
