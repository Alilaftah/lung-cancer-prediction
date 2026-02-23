import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# استيراد المكتبات الخاصة بالمعالجة والتعلم الآلي حسب الخطة
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1) & 2) & 3) تحميل وتنظيف وتحويل البيانات (حسب كودك وخطة العمل)
# =========================================================
def clean_and_prepare_data(file_path):
    # الخطوة 1: تحميل الملف
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the file name and path.")

    df = pd.read_csv(file_path)
    print(f"Original data size: {df.shape}")

    # الخطوة 2: حذف التكرارات والقيم الناقصة
    if df.isnull().values.any():
        df.dropna(inplace=True)
        print("- Missing values deleted.")

    if df.duplicated().any():
        df.drop_duplicates(inplace=True)
        print("- Duplicate rows deleted.")

    # تنظيف المسافات الزائدة في النصوص
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # الخطوة 3: تحويل النصوص إلى أرقام (Label Encoding) لكل الأعمدة غير الرقمية
    le = LabelEncoder()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"- Encoding column: {col}")
            df[col] = le.fit_transform(df[col].astype(str))
    
    print("- Encoding completed (Categorical data converted to numbers).")
    return df

# تنفيذ الجزء الأول
file_name = 'survey lung cancer 1.csv' # تأكد أن الملف بهذا الاسم في نفس مجلد الكود
data_ready = clean_and_prepare_data(file_name)

# =========================================================
# 4) استعراض جودة البيانات
# =========================================================
print("\n--- 4) Data Info ---")
print(data_ready.info())

# =========================================================
# 5) تحديد الميزات (X) والهدف (y)
# =========================================================
X = data_ready.drop('LUNG_CANCER', axis=1)
y = data_ready['LUNG_CANCER']

# =========================================================
# 6) تقسيم البيانات (80% تدريب، 20% اختبار)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# =========================================================
# 7) موازنة البيانات باستخدام ADASYN (مهم جداً طبياً)
# =========================================================
# توليد عينات ذكية للفئة الأقل لضمان عدم انحياز النموذج
adasyn = ADASYN(random_state=88)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
print(f"\n- 7) ADASYN: Training data balanced to {X_train_res.shape}")

# =========================================================
# 8) التقييس (Standardization)
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
print("- 8) Standardization applied.")

# =========================================================
# 9) تقليل الأبعاد (PCA) - الحفاظ على 75% من التباين
# =========================================================
pca = PCA(n_components=0.75)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"- 9) PCA: Features reduced to {X_train_pca.shape[1]} components (75% variance).")
# =========================================================
# 10) تدريب وتقييم كل خوارزمية على حدة
# =========================================================
print("\n--- 10) Individual Algorithms Evaluation ---")

individual_models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest (RF)": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (XGB)": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Gradient Boosting (GB)": GradientBoostingClassifier(random_state=42),
    "Gaussian Naive Bayes (GNB)": GaussianNB()
}

for name, model in individual_models.items():
    print(f"\nEvaluating: {name}")
    model.fit(X_train_pca, y_train_res)
    predictions = model.predict(X_test_pca)
    print(f"Accuracy of {name}: {accuracy_score(y_test, predictions):.2%}")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, predictions))
    print("-" * 30)

# =========================================================
# 11) تدريب النموذج (Stacking Ensemble) والتقييم النهائي
# =========================================================
# تعريف النماذج الأساسية الأربعة
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('gnb', GaussianNB())
]

# استخدام Stacking لدمجهم معاً لرفع الدقة
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
)

print("\n- 11) Training Stacking Ensemble Model (KNN, RF, XGB, GB)...")
stacking_clf.fit(X_train_pca, y_train_res)

# التنبؤ والنتائج
y_pred = stacking_clf.predict(X_test_pca)

print("\n" + "="*40)
print(f" FINAL STACKING MODEL ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
print("="*40)

print("\n--- Stacking Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# رسم مصفوفة الارتباك للتقرير النهائي للمشروع
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Lung Cancer Prediction (Stacking)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()