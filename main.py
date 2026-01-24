import pandas as pd
import numpy as np
import os
# Fixing the ImportError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"දත්ත ගොනුව හමු නොවීය: {path}")


def clean_data(df):
    # Numerical columns පමණක් තෝරාගෙන පිරිසිදු කිරීම
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Smart Imputation while preserving correlations
    imputer = IterativeImputer(random_state=42)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # සෘණ අගයන් ලැබුණහොත් ඒවා ඉවත් කරමු (Post-processing)
    for col in ['Daily_Steps', 'Workout_Min', 'Calories_Burned']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def feature_engineering(df):
    # පියවර ගණන අනුව Activity Level වර්ගීකරණය
    # Labels ඉංග්‍රීසියෙන් තැබීම Professional (README එකට ගැලපෙන ලෙස)
    df['Activity_Level'] = pd.cut(df['Daily_Steps'],
                                  bins=[0, 5000, 10000, 15000, 30000],
                                  labels=['Sedentary', 'Moderate', 'Active', 'Very Active'])
    return df


def main():
    print("Aura Fitness Data Pipeline Started... 🚀")

    # 22 හැවිරිදි IT ශිෂ්‍යයෙකු ලෙස නිවැරදි File Paths පාවිච්චි කරමු
    input_file = './data/aura_fitness_final.csv'
    output_file = './data/aura_fitness_cleaned.csv'

    try:
        df = load_data(input_file)
        print("1. Data Loaded Successfully. ✅")

        df = clean_data(df)
        print("2. Data Cleaning (Iterative Imputation) Done. ✅")

        df = feature_engineering(df)
        print("3. Feature Engineering Completed. ✅")

        df.to_csv(output_file, index=False)
        print(f"4. Process Completed! Cleaned data saved to: {output_file} 🎉")

    except Exception as e:
        print(f"Error එකක් සිදු විය: {e}")


if __name__ == "__main__":
    main()