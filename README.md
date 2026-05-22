# Shopee Aspect-Based Sentiment Analysis using Traditional Machine Learning

This project focuses on Aspect-Based Sentiment Analysis (ABSA) for Vietnamese Shopee product reviews.

## Task

Given a customer review comment, the system predicts sentiment labels for four aspects:

1. Product quality (`chat_luong`)
2. Packaging (`dong_goi`)
3. Delivery (`van_chuyen`)
4. Price (`gia_ca`)

Each aspect has three possible sentiment labels:

1. **Data Collection:** Collect around 2,000 comments from Shopee, focusing on a specific domain (e.g., cosmetics, electronics, or fashion) to ensure data consistency.
2. **Define Aspect List:** The actual dataset categorizes comments into 5 main aspects:
   - `chat_luong` (Product Quality)
   - `dong_goi` (Packaging)
   - `van_chuyen` (Shipping/Delivery)
   - `gia_ca` (Price)
   - `overall` (Overall experience)
3. **Data Labeling:** Each comment will be split into multiple data samples corresponding to each aspect.
   - Format: `(comment, aspect, sentiment)`
   - Sentiment: `positive`, `neutral`, or `negative`. The data also features a `text_input` column combining the cleaned sentence and the aspect.
4. **Text Preprocessing:**
   - Convert text to lowercase.
   - Remove punctuation and special characters.
   - Tokenization and stopword removal (using the `underthesea` library for Vietnamese).
5. **Feature Extraction:**
   - Convert text into numerical data using **TF-IDF**.
   - Incorporate n-grams (bigrams) to capture meaningful phrases like "không tốt" (not good) or "rất đẹp" (very beautiful).

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
