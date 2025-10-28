import pandas as pd
import os

def main():

    input_path = "data/raw/telco_churn.csv"
    output_path = "outputs/telco_churn_clean.csv"

    df = pd.read_csv(input_path)

    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(columns=['customer_id'])

    for col in ['total_charges', 'monthly_charges', 'tenure_months', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    col_numericas = df.select_dtypes(include=['number']).columns.tolist()
    for col in col_numericas:
        df[col] = df[col].fillna(df[col].median())

    col_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in col_categoricas:
        df[col] = df[col].fillna('Unknown')

    df['phone_service'] = df['phone_service'].map({'Yes':1,'No':0, 'No phone service':0})
    
    df = pd.get_dummies(df, columns=col_categoricas, drop_first=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()