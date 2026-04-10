# Laptop Price Prediction using Machine Learning

## Overview

This project focuses on predicting laptop prices using Machine Learning techniques. The dataset contains various laptop specifications such as brand, processor details, RAM, storage, and other features. The goal is to build a model that can accurately estimate the price based on these features.

---

## Dataset Information

* Total Records: **823**
* Features: **19 columns**
* Target Variable: **Price**

### Key Features:

* Brand
* Processor (brand, name, generation)
* RAM (size & type)
* Storage (SSD + HDD)
* Operating System
* Graphics Card
* Warranty, Touchscreen, MS Office
* Ratings and Reviews

---

## Data Preprocessing

### 1. Data Cleaning

* Removed unnecessary column: `weight`
* Converted string values to numeric:

  * `ram_gb`, `ssd`, `hdd`, `graphic_card_gb`
* Cleaned `rating` column and later dropped it
* Converted categorical binary features:

  * `Touchscreen`, `msoffice` → 0/1
* Converted `warranty`:

  * "No warranty" → 0
  * Others → 1

---

### 2. Feature Engineering

* Created new feature:

  * **Total_storage = SSD + HDD**
* Simplified processor name:

  * Example: "Core i5" → "Core"
* Converted processor generation to numeric
* Handled missing values like "Not Available" → 0

---

### 3. Feature Selection

* Input Features (X): All columns except `Price`
* Target (Y): `Price`

---

## Model Building

### Train-Test Split

* 80% Training
* 20% Testing
* Random State: 42

---

### Encoding

Used **OneHotEncoder** for categorical variables:

* brand
* processor_brand
* processor_name
* ram_type
* os
* os_bit

Implemented using **ColumnTransformer + Pipeline**

---

## Models Used

### 1. Linear Regression

* Simple baseline model

**Results:**

* MAE: 19035
* RMSE: 29575

---

### 2. Random Forest Regressor (Improved Model)

* Used 200 trees (`n_estimators=200`)

**Results:**

* MAE: 13834
* RMSE: 25997

Improvement shows better handling of non-linear relationships.

---

## Model Evaluation Metrics

* **MAE (Mean Absolute Error)**
  Average difference between actual and predicted price

* **RMSE (Root Mean Squared Error)**
  Penalizes large errors more than MAE

---

## Sample Predictions

| Actual Price | Predicted Price |
| ------------ | --------------- |
| 104990       | 119541          |
| 41890        | 53595           |
| 57500        | 73257           |

---

## Visualization

* Scatter Plot between **Actual vs Predicted Price**
* Helps visualize model performance and error distribution

---

## Single Prediction Example

Model can also predict price for a single laptop:

```python
one = X_test.iloc[0:1]
pipe.predict(one)
```

---

## Key Learnings

* Feature engineering plays a major role in improving model performance
* Random Forest performs better than Linear Regression for this dataset
* Pipelines help in clean and reusable ML workflows
* Encoding categorical data properly is crucial

---

## Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Try advanced models:

  * XGBoost
  * LightGBM
* Feature selection techniques
* Outlier handling
* Cross-validation

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib

---

## Conclusion

This project successfully builds a machine learning pipeline to predict laptop prices with decent accuracy. The Random Forest model significantly improves prediction performance compared to Linear Regression.

---

## Author

Shubham Thakur

