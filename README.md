# **House Price Prediction (Regression)**

Predicting house prices using the **Kaggle House Prices Dataset**.  
This project performs **EDA**, **data cleaning**, **data preprocessing**, and trains **three ML models** to estimate home values.

---

## **Dataset**
Kaggle: *House Prices – Advanced Regression Techniques*  
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

---

## **Exploratory Data Analysis (EDA)**

The notebook (`notebook.ipynb`) includes:

### Basic analysis
- `.info()` for structure  
- `.describe()` for statistics  
- Missing value inspection  
- Removing columns with **>40% missing**
- And more..

### Target variable (SalePrice) analysis
- Distribution plot  
- Skewness check

### Feature relationships
- Correlation matrix  
- Top correlated features with SalePrice  
- Scatter plots (e.g., **GrLivArea vs SalePrice**)  
- Boxplot (**OverallQual vs SalePrice**)  

---

## **Data Preprocessing**
Steps applied before modeling:

1. **Remove columns** with more than **40% missing values**  
2. **Fill remaining missing values**  
   - Numerical -> median  
   - Categorical -> mode  
3. **Drop `Id` column**  
4. **One-hot encode categorical features**  
5. **Train/validation split (80/20)**  

All preprocessing logic is implemented in `src/utils.py`.

---

## **🤖 Models Trained**
Three regression models were trained for comparison:

| Model | RMSE (Validation) |
|-------|-------------------|
| **Gradient Boosting Regressor** | ~28,455 |
| **Random Forest Regressor**     | ~28,470 |
| **Linear Regression**           | ~51,970 |

The model performed the best is saved as the final model.
* Note: These results are from a random run, GB and RF are generally close in performance on this specific dataset with the current setup
---

## **Feature Importance**
A bar plot of the **top 10 most important features** is included in the notebook.

Major contributors include:
- OverallQual: Overall material and finish quality (a rating from 1-10)
- GrLivArea: Living area above ground
- TotalBsmtSF: Total basement area (sq ft)

---

## **Saving the Best Model**
The best model is saved as:
`models/house_price_model.pkl`

Load it using:
`
import joblib
model = joblib.load("models/house_price_model.pkl")`

## Project Structure

house-price-prediction/
├── README.md
├── notebook.ipynb
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── house_price_model.pkl
└── src/
    ├── train.py
    └── utils.py
