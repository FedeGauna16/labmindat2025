import pandas as pd
import os

def main():

    input_path = "data/raw/telco_churn.csv"
    output_path = "outputs/telco_churn_clean.csv"

    df = pd.read_csv(input_path)

    df = df.drop_duplicates()
    df = df.dropna()


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset limpio guardado en {output_path}")


if __name__ == "__main__":
    main()