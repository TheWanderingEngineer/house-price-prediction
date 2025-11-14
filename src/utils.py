import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load CSV file and return DataFrame."""
    return pd.read_csv(path)
    ad_csv(path)

 
def remove_high_missing(df, threshold=0.40):
    """Drop columns with missing ratio > threshold."""
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=cols_to_drop)


def fill_missing(df):
    """Fill missing numeric with median, categorical with mode."""
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_and_split(df, target_col="SalePrice", test_size=0.2, random_state=42):
    """Drop Id, encode categoricals, perform train/valid split."""
    
    # drop Id if exists
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    
    # split target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # one-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_valid, y_train, y_valid
