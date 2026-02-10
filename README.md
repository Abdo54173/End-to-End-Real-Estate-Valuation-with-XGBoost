# 🏠 End-to-End Real Estate Valuation with XGBoost

An end-to-end machine learning pipeline for predicting housing prices using the Ames Housing dataset.  
This project demonstrates structured ML engineering: data preprocessing, feature engineering, regularization, boosting, and model tuning.

---

# 📌 Problem Statement

- 🎯 Task: 'Regression'
- 📊 Target: 'SalePrice'
- 📏 Evaluation Metric: 'RMSE' on 'log(SalePrice)'
- 🧠 Goal: Build a stable and optimized model without over-engineering

---

# 🛠️ Project Pipeline

## 1️⃣ Target Processing

- Applied `'log1p' transformation` to reduce skewness
- Stabilized variance for better regression performance

---

## 2️⃣ Data Cleaning

- Context-aware missing value imputation
- Logical filling (e.g., "None" for non-existing features)
- Group-wise median filling for structured features (e.g., LotFrontage)

---

## 3️⃣ Feature Engineering

Created meaningful features:

- `TotalSF`
- `HouseAge`
- `RemodAge`
- `TotalBath`
- `OverallScore`
- `Quality × Size interactions`
- Selected polynomial interactions (controlled, not brute force)

---

## 4️⃣ Encoding Strategy

- ✔️ 'Ordinal Encoding' for ordered features (quality-related)
- ✔️ 'One-Hot Encoding' for nominal features

---

## 5️⃣ Linear Models

| Model | CV RMSE |
|--------|----------|
| Linear Regression | 0.139 |
| Ridge (tuned) | 0.129 |
| Lasso (tuned) | 0.1247 |
| Lasso + Deep Feature Engineering | 0.1243 |

🔎 Observation:
'L1 Regularization' significantly improved performance by removing noisy and redundant features (268 → 75 active features).

---

## 6️⃣ Tree-Based Models

### 🔹 XGBoost (Baseline)

- RMSE ≈ 0.1212

### 🔹 XGBoost (Tuned max_depth)

- max_depth = 3
- RMSE ≈ 0.1189

### 🔹 Final Tuned XGBoost

Best configuration:

- max_depth = 3
- learning_rate = 0.02
- n_estimators = 3000
- subsample = 0.8
- colsample_bytree = 0.8
- reg_alpha = 0.1
- reg_lambda = 1

🏆 Final CV Score:
'RMSE = 0.1183'

---

# 📊 Final Model

### 🌳 XGBoost Regressor

Why this model?

- Captures non-linearity automatically
- Handles feature interactions internally
- Strong bias-variance balance
- Minimal overfitting with proper tuning

This model outperformed all linear alternatives.

---

# 📈 Performance Progression

| Stage | RMSE |
|--------|--------|
| Baseline Linear | 0.139 |
| Regularized Linear | 0.124 |
| Tuned XGBoost | **0.1183** |

📉 Significant improvement through structured engineering, not brute-force tuning.

---

# 🧠 Key Learnings

- Feature representation matters more than model complexity.
- Regularization (L1 vs L2) dramatically affects performance.
- Tree models reduce need for manual non-linearity.
- Over-engineering gives diminishing returns.
- Clean pipeline > Kaggle tricks.

---

# 🚀 Final Submission Workflow

1. Train final XGBoost on full training set
2. Predict on test set (log scale)
3. Apply `expm1` to revert transformation
4. Export submission file

---

# 📂 Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost

---

# 🏁 Conclusion

This project demonstrates a structured, production-ready ML workflow:

✔ Data Cleaning  
✔ Feature Engineering  
✔ Regularization  
✔ Boosting  
✔ Hyperparameter Tuning  
✔ Bias-Variance Control  

Final Result:
'RMSE ≈ 0.1183'

A balanced, interpretable, and optimized regression pipeline.

---

# 👑 Author

Built as a complete end-to-end ML engineering case study.
