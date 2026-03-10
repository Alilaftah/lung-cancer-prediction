import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

class ModelManager:
    def __init__(self):
        # تهيئة المتغيرات الأساسية للنموذج والبيانات
        self.model = None
        self.scaler = None
        self.pca = None
        self.label_encoders = {}
        self.file_path = 'survey lung cancer 1.csv'
        self.model_file = 'lung_cancer_model.joblib'

    def clean_and_prepare_data(self, df=None):
        # تحميل البيانات إذا لم يتم تمريرها كمعامل
        if df is None:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            df = pd.read_csv(self.file_path)

        # التنظيف الأساسي: إزالة القيم الفارغة والتكرارات
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        # إزالة المسافات الزائدة من النصوص
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # تحويل البيانات الفئوية إلى عددية
        temp_le = {}
        processed_df = df.copy()
        for col in processed_df.columns:
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                temp_le[col] = le
        
        # --- المرحلة 3: هندسة الميزات (حسب خارطة الطريق) ---
        # دمج القلق واصفرار الأصابع لإنشاء ميزة جديدة
        if 'ANXIETY' in processed_df.columns and 'YELLOW_FINGERS' in processed_df.columns:
            processed_df['ANX_YEL_FIN'] = processed_df['ANXIETY'] * processed_df['YELLOW_FINGERS']
        
        self.label_encoders = temp_le
        return processed_df

    def generate_eda_plots(self, df):
        # إنشاء مجلد الرسوم البيانية إذا لم يكن موجوداً
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        plt.style.use('ggplot')
        
        # 1. مصفوفة ارتباط بيرسون مع تعليق توضيحي
        plt.figure(figsize=(14, 11))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Pearson Correlation Matrix (Feature Relationship Analysis)', fontsize=15)
        plt.figtext(0.5, 0.01, "Figure 1: Pearson Correlation Matrix to analyze linear relationships between healthcare indicators and Lung Cancer.", 
                    ha="center", fontsize=11, fontweight='bold', bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.savefig('plots/correlation_matrix.png', bbox_inches='tight')
        plt.close()

        # 2. توزيع الحالات حسب العمر مع تعليق توضيحي
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df, x='AGE', hue='LUNG_CANCER', multiple='stack', bins=15, palette='viridis')
        plt.title('Lung Cancer Cases Distribution by Age Group', fontsize=16)
        plt.xlabel('Patient Age', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.figtext(0.5, -0.05, "Figure 2: Demographic distribution of observed Lung Cancer cases across different age intervals.", 
                    ha="center", fontsize=11, fontweight='bold', bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.savefig('plots/age_distribution.png', bbox_inches='tight')
        plt.close()

        # 3. توزيع المتغير المستهدف مع تعليق توضيحي علمي
        plt.figure(figsize=(8, 8))
        counts = df['LUNG_CANCER'].value_counts()
        labels = [f'NO ({counts[0]})', f'YES ({counts[1]})' if len(counts)>1 else f'NO ({counts[0]})']
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], 
                startangle=140, explode=(0.05, 0))
        plt.title('Distribution of the Target Variable (LUNG_CANCER Signal)', fontsize=14)
        plt.figtext(0.5, 0.01, "Figure 3: Class balance representation indicating the distribution of positive and negative diagnosis records.", 
                    ha="center", fontsize=11, fontweight='bold', bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.savefig('plots/target_distribution.png', bbox_inches='tight')
        plt.close()

        print("--- Scientific EDA plots generated in 'plots' folder ---")

    def train_full_pipeline(self):
        # تحميل وتجهيز البيانات
        df = self.clean_and_prepare_data()
        
        # إنشاء الرسوم البيانية الاستكشافية
        self.generate_eda_plots(df)
        
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']

        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # موازنة البيانات باستخدام تقنية ADASYN
        adasyn = ADASYN(random_state=88)
        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

        # التقييس (Standardization)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)

        # تقليل الأبعاد باستخدام PCA
        self.pca = PCA(n_components=0.75)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        # النماذج الأساسية للتجميع (Stacking)
        base_models = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('gnb', GaussianNB())
        ]

        # إنشاء مصنف التجميع (Stacking Classifier)
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
        )

        # تدريب النموذج
        self.model.fit(X_train_pca, y_train_res)
        
        # تقييم النماذج الفردية لإعداد التقارير
        reports = []
        for name, m in base_models:
            m.fit(X_train_pca, y_train_res)
            preds = m.predict(X_test_pca)
            acc = accuracy_score(y_test, preds)
            reports.append(f"Algorithm: {name.upper()}\nAccuracy: {acc:.2%}\n{classification_report(y_test, preds)}")

        # التقييم النهائي لنموذج التجميع (Stacking Model)
        stack_preds = self.model.predict(X_test_pca)
        stack_acc = accuracy_score(y_test, stack_preds)
        reports.append(f"FINAL STACKING MODEL\nFinal Accuracy: {stack_acc:.2%}\n{classification_report(y_test, stack_preds)}")

        return "\n".join(reports), X_test_pca, y_test

    def save_model(self):
        # حفظ النموذج والكائنات المرتبطة به
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoders': self.label_encoders
        }
        joblib.dump(data, self.model_file)
        print(f"Model saved successfully to {self.model_file}")

    def load_model(self):
        # تحميل النموذج المحفوظ إذا كان موجوداً
        if os.path.exists(self.model_file):
            data = joblib.load(self.model_file)
            self.model = data['model']
            self.scaler = data['scaler']
            self.pca = data['pca']
            self.label_encoders = data['label_encoders']
            return True
        return False

    def predict(self, raw_input_dict):
        # إضافة الميزة الهندسية للمدخل الجديد
        if 'ANXIETY' in raw_input_dict and 'YELLOW_FINGERS' in raw_input_dict:
            raw_input_dict['ANX_YEL_FIN'] = raw_input_dict['ANXIETY'] * raw_input_dict['YELLOW_FINGERS']
            
        df_input = pd.DataFrame([raw_input_dict])
        
        # تطبيق التقييس و PCA
        scaled = self.scaler.transform(df_input)
        pca_data = self.pca.transform(scaled)
        
        # التنبؤ
        prediction = self.model.predict(pca_data)[0]
        return prediction
