# Aspect-Based Sentiment Analysis (ABSA) for Social Media Comments

🎯 **Project Objective**
The goal of this project is to build a system capable of analyzing user sentiment on social media platforms (Shopee), focusing on specific aspects (aspect-based) rather than evaluating the sentiment of the entire sentence.

Example: *"Giao hàng nhanh nhưng sản phẩm kém chất lượng"* (Fast delivery but poor product quality)
- Delivery (`giao_hang`): **Positive**
- Quality (`chat_luong`): **Negative**

This project utilizes the **Logistic Regression** model combined with basic Natural Language Processing (NLP) techniques.

---

## 🧩 Phase 1: Data Collection and Preparation

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

---

## 🤖 Phase 2: Model Building

- Use the **Logistic Regression** model (implemented via `scikit-learn`) to predict sentiment for each `(comment, aspect)` pair.
- **Input Features:** Combine the comment and the aspect into the input, for example: *"Sản phẩm đẹp nhưng giá cao [ASPECT=gia_ca]"* (Beautiful product but high price [ASPECT=price]).
- This approach helps the model understand that the same sentence can yield different sentiment results when considering different aspects.

---

## ⚙️ Phase 3: Model Improvement and Optimization

1. **Regularization:** Apply L2 Regularization (default in Logistic Regression) to prevent overfitting, helping the model generalize better on unseen data.
2. **Baseline Model:** Compare the performance of Logistic Regression with a simpler baseline model like **Naive Bayes** (`MultinomialNB`) to demonstrate the effectiveness of the proposed method.
3. **Experiments:**
   - Compare the performance between using unigrams and bigrams.
   - Evaluate the effectiveness of including the aspect in the input versus not using the aspect.

---

## 📊 Phase 4: Model Evaluation

1. **Data Splitting:** The data is divided into a Train set, Validation set (for parameter tuning), and Test set (for final evaluation). Cross-validation can be used if the dataset is small.
2. **Evaluation Metrics:**
   - Accuracy, Precision, Recall.
   - **F1-score:** The most important metric to balance between precision and recall.
3. **Result Analysis:**
   - Use a Confusion Matrix to observe and analyze misclassified cases (e.g., sentences with multiple aspects, ambiguous sentences, or sentences containing negation words).

---

## ✅ Conclusion

This project delivers a simple yet effective ABSA system, suitable for the scope of the Machine Learning 1 course. Through this project, the team will:
- Understand the text data processing pipeline.
- Proficiently apply Machine Learning models to real-world problems.
- Systematically analyze and evaluate model performance.
- Elevate basic Sentiment Analysis to a higher level by effectively handling aspect-based inputs while maintaining feasibility during implementation.
