import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
import os

def main():
    # 1. تحميل الملف الأصلي
    if not os.path.exists('survey lung cancer 1.csv'):
        print("File not found.")
        return
        
    df = pd.read_csv('survey lung cancer 1.csv')
    
    # 2. المعالجة المسبقة والتنظيف (بشكل صارم)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # 3. ترميز رقمي قوي
    df_numeric = df.copy()
    
    # ترميز المتغير المستهدف
    df_numeric['LUNG_CANCER'] = df_numeric['LUNG_CANCER'].astype(str).str.strip().map({'YES': 1, 'NO': 0})
    
    # ترميز الجنس
    df_numeric['GENDER'] = df_numeric['GENDER'].astype(str).str.strip().map({'M': 1, 'F': 0})
    
    # ترميز الأعراض (نعم=2، لا=1 لمطابقة المنطق الأصلي في الورقة العلمية)
    for col in df_numeric.columns:
        if col not in ['GENDER', 'AGE', 'LUNG_CANCER']:
            df_numeric[col] = df_numeric[col].astype(str).str.strip().map({'YES': 2, 'NO': 1, '2': 2, '1': 1})
            # إذا فشل التحويل، نضع القيمة الافتراضية 1 (لا)
            df_numeric[col] = df_numeric[col].fillna(1).astype(int)

    # التحقق النهائي من وجود قيم فارغة
    df_numeric.dropna(inplace=True)
    
    X = df_numeric.drop('LUNG_CANCER', axis=1)
    y = df_numeric['LUNG_CANCER']
    
    # 4. تطبيق تقنية ADASYN
    # تتطلب ADASYN الفئات 0 و 1. بما أن y تحتوي على 0 و 1، فهي جاهزة.
    
    adasyn = ADASYN(random_state=88)
    X_res, y_res = adasyn.fit_resample(X, y)
    
    df_balanced = pd.concat([X_res, y_res], axis=1)
    
    # 5. إزالة أي تكرارات قد تنتج عن ADASYN بسبب طبيعة البيانات المنفصلة
    df_final = df_balanced.drop_duplicates()
    
    # تنظيف نهائي للأسماء (اختياري)
    df_final.to_csv('survey_lung_cancer_cleaned.csv', index=False)
    
    print("--- Scientific & Clean Balancing ---")
    print(f"Final Records (Unique & Balanced): {len(df_final)}")
    print("Class Distribution:")
    print(df_final['LUNG_CANCER'].value_counts())
    print(f"Duplicates: {df_final.duplicated().sum()}")
    print(f"Missing Values: {df_final.isnull().sum().sum()}")

if __name__ == "__main__":
    main()
