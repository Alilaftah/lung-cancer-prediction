import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
import os

def main():
    # 1. Load original
    if not os.path.exists('survey lung cancer 1.csv'):
        print("File not found.")
        return
        
    df = pd.read_csv('survey lung cancer 1.csv')
    
    # 2. Pre-process and Clean (STRICT)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # 3. Robust Numeric Encoding
    df_numeric = df.copy()
    
    # Target encoding
    df_numeric['LUNG_CANCER'] = df_numeric['LUNG_CANCER'].astype(str).str.strip().map({'YES': 1, 'NO': 0})
    
    # Gender encoding
    df_numeric['GENDER'] = df_numeric['GENDER'].astype(str).str.strip().map({'M': 1, 'F': 0})
    
    # Symptoms encoding (YES=2, NO=1 to match original paper logic)
    for col in df_numeric.columns:
        if col not in ['GENDER', 'AGE', 'LUNG_CANCER']:
            df_numeric[col] = df_numeric[col].astype(str).str.strip().map({'YES': 2, 'NO': 1, '2': 2, '1': 1})
            # If any mapping failed (resulted in NaN), we fill with 1 (NO) as default
            df_numeric[col] = df_numeric[col].fillna(1).astype(int)

    # Final check for any NaNs
    df_numeric.dropna(inplace=True)
    
    X = df_numeric.drop('LUNG_CANCER', axis=1)
    y = df_numeric['LUNG_CANCER']
    
    # 4. Apply ADASYN
    # ADASYN requires class 0 and 1. If we have 1 and 2, it might need adjustment 
    # but here y is 0 and 1.
    
    adasyn = ADASYN(random_state=88)
    X_res, y_res = adasyn.fit_resample(X, y)
    
    df_balanced = pd.concat([X_res, y_res], axis=1)
    
    # 5. Remove any duplicates that ADASYN might have created due to discrete data
    df_final = df_balanced.drop_duplicates()
    
    # Final cleanup of names (Optional but good)
    df_final.to_csv('survey_lung_cancer_cleaned.csv', index=False)
    
    print("--- Scientific & Clean Balancing ---")
    print(f"Final Records (Unique & Balanced): {len(df_final)}")
    print("Class Distribution:")
    print(df_final['LUNG_CANCER'].value_counts())
    print(f"Duplicates: {df_final.duplicated().sum()}")
    print(f"Missing Values: {df_final.isnull().sum().sum()}")

if __name__ == "__main__":
    main()
