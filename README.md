# 🛡️ Health Insurance Fraud Detection Pipeline

An end-to-end Machine Learning classification pipeline designed to identify fraudulent policyholders and mitigate financial exposure for health insurance providers.

This project demonstrates a production-grade data science workflow: joining 4 relational database tables, engineering behavioral features with SQL CTEs and Window Functions, building a leak-free multi-branch Scikit-Learn `ColumnTransformer` with domain-specific imputations, optimizing an `XGBClassifier` via 5-fold cross-validated hyperparameter tuning (`RandomizedSearchCV`), and diagnosing model convergence with learning curves.

---

## 🛠️ Tech Stack
* **Database & SQL:** `SQLite3`, Advanced SQL (Common Table Expressions `WITH`, Window Functions `OVER (PARTITION BY)`, Aggregations, `CASE WHEN`)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Machine Learning & Pipeline:** `Scikit-Learn` (`Pipeline`, `ColumnTransformer`, `RobustScaler`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`, `SimpleImputer`, `RandomizedSearchCV`, `learning_curve`)
* **Model Engine:** `XGBoost` (`XGBClassifier`)
* **Evaluation & Diagnostics:** Generalization Gap, Learning Curves, Feature Importances, Probability Threshold Calibration (`predict_proba`)

---

## 📊 Project Architecture

### 1. Relational Data Extraction & SQL Feature Engineering
The database (`insurance_fraud.db`) models an insurance ecosystem across 4 tables: `agents`, `policyholders`, `claims`, and `fraud_labels`.

To resolve the **1-to-many relationship** between policyholders and their multiple claim records without row duplication or data leakage:
* **Common Table Expression (CTE `ClaimStats`):** Aggregated claim events into a single row per policyholder:
  * `total_claims`: Frequency and velocity of claims filed.
  * `total_claim_amount`: Cumulative financial exposure.
  * `avg_claim_amount`: Average claim severity profile.
  * `emergency_count`: High-velocity emergency claims (`SUM(CASE WHEN claim_type = 'Emergency' THEN 1 ELSE 0 END)`).
  * `max_hospitalization`: Peak hospital stay duration (`MAX(hospitalization_days)`).
  * `claim_type_variety`: Behavioral entropy and claim category diversity (`COUNT(DISTINCT claim_type)`).
* **Window Function (`avg_region_claims`):** Calculated regional baseline benchmarks (`AVG(total_claim_amount) OVER (PARTITION BY region)`) to normalize local cost-of-living differences.
* **Anchor Join:** Executed `LEFT JOIN` on `policyholders` to preserve policyholders with zero claims (600 unique entities).

### 2. Leak-Free Preprocessing Pipeline (Scikit-Learn `ColumnTransformer`)
Different data modalities require tailored statistical treatments. We constructed a 6-branch pipeline:
1. **Financial Outliers (`total_claim_amount`, `avg_claim_amount`):** Policyholders with 0 claims are imputed with `fill_value=0`. Scaled using `RobustScaler` (median & IQR) to handle extreme financial blowouts up to ₦544,121 without mean distortion.
2. **Emergency Velocity (`emergency_count`):** Non-claimants imputed with `fill_value=0`, followed by `StandardScaler`.
3. **Informative Missingness (`smoker`):** Refusal to disclose smoking status on insurance applications contains predictive signal. Imputed with `fill_value='Unknown'` and one-hot encoded, allowing the model to capture undisclosed risk.
4. **Ordered Socioeconomic Hierarchy (`income_bracket`):** Natural ordinal hierarchy (`Low < Medium < High < Very High`), encoded via `OrdinalEncoder`.
5. **Nominal Demographics (`gender`, `region`, `agency_type`):** Imputed with `most_frequent` and encoded via `OneHotEncoder(drop='first', sparse_output=False)`.
6. **Standard Numericals (`age`, `bmi`, `years_experience`, `total_claims`, `max_hospitalization`, `claim_type_variety`, `avg_region_claims`):** Imputed with `median` and scaled with `StandardScaler`.
7. **Stratified Split:** Executed `train_test_split(..., stratify=y)` to guarantee the exact 85% clean / 15% fraud class distribution across training and testing partitions.

### 3. Model Training & Hyperparameter Tuning (`XGBClassifier`)
* Built an encapsulated `Pipeline([('preprocessor', processor), ('model', XGBClassifier(random_state=42))])` to guarantee zero data leakage during cross-validation.
* Configured a 5-fold cross-validated `RandomizedSearchCV` tuning `max_depth`, `n_estimators`, and `learning_rate`.
* **Performance:** Achieved **85.0% Cross-Validation Accuracy** and **85.0% Holdout Test Accuracy**.

### 4. Learning Curve Diagnosis & Convergence
Evaluated training vs. validation accuracy across 10 progressive training sizes (38 to 384 samples):
* **Initial Size (38 rows):** Train Mean = 0.7737, Val Mean = 0.8167 (Gap = -0.0430).
* **Final Capacity (384 rows):** Train Mean = **0.8500**, Val Mean = **0.8500**, **Generalization Gap = 0.0000**!
* **Conclusion:** The model reached perfect bias-variance equilibrium with zero overfitting. The validation curve plateaus at 85%, confirming that gathering additional rows is not the bottleneck—gains require probability threshold calibration.

### 5. Feature Importance & Top Drivers
Extracted transformed feature names from the preprocessor step and paired them with XGBoost `feature_importances_`:
1. `outlier__total_claim_amount` (0.0954) — Financial magnitude exposure
2. `emerg__emergency_count` (0.0854) — Emergency claim frequency
3. `nums__age` (0.0823) — Demographic risk profile
4. `cat__agency_type_Exclusive` (0.0731) — Agency origination channel
5. `nums__bmi` (0.0701) — Physical health indicator
6. `smoke__smoker_Unknown` (0.0107) — Undisclosed smoker status carried statistically significant predictive weight

---

## 💼 Executive Business Insights & Decision Strategy

1. **Financial Recovery Potential:**  
   In a portfolio with ~150 fraudulent claims averaging ₦5,000,000 each (₦750,000,000 total exposure), catching 50% to 60% of cases recovers **over ₦375,000,000 to ₦450,000,000** in unrecoverable cash.

2. **Investigator Workload Triage (50 Audits / Month):**  
   Human forensic investigators cannot audit all claims manually. Rather than random sampling, we use `model.predict_proba(X_test)[:, 1]` to rank all claimants by fraud probability descending, handing investigators the Top 50 highest-confidence cases.

3. **Asymmetric Error Costs:**  
   Missing fraud (False Negative) costs ₦5,000,000 in lost capital. A false alarm (False Positive) costs 10 minutes of an investigator's time to verify a hospital receipt. It is far better to have false alarms than missed fraud, because manual audits clear innocent policyholders.

4. **Critique of the CEO's ₦300,000 Rule:**  
   * **Massive False Positives:** Legitimate policyholders with severe illnesses (cardiac surgery, cancer treatments) exceed ₦300,000. Calling them fraudsters triggers customer churn, lawsuits, and regulatory penalties.
   * **Massive False Negatives:** Fraud rings exploit simple thresholds by submitting multiple small claims (e.g., four ₦80,000 claims totaling ₦320,000). The CEO's rule misses them entirely, whereas our model's `total_claims` and `emergency_count` catches them immediately.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd insurance-fraud-detection
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Insurance_Fraud_Detection.ipynb
   ```
3. Run all cells to inspect the SQL feature extraction, preprocessing pipeline, model tuning, learning curves, and feature importance rankings.
