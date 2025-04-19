import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y/%m")
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df.to_csv(output_path, index=False)
    return df