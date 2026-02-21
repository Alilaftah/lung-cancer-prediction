# Lung Cancer Prediction Project | مشروع التنبؤ بسرطان الرئة

## Project Overview | نظرة عامة على المشروع
This project aim to predict lung cancer using machine learning techniques. It utilizes a dataset of patient surveys to identify potential cases based on various health indicators and symptoms.

يهدف هذا المشروع إلى التنبؤ بالإصابة بسرطان الرئة باستخدام تقنيات تعلم الآلة. يعتمد المشروع على مجموعة بيانات مستخلصة من استبيانات المرضى لتحديد الحالات المحتملة بناءً على مؤشرات صحية وأعراض متنوعة.

## Features | المميزات
- **Data Cleaning**: Handles missing values and duplicates.
- **Data Preprocessing**: Label encoding for categorical data and standard scaling for numerical features.
- **SMOTE/ADASYN**: Balances the dataset to improve model accuracy for minority classes.
- **Dimensionality Reduction**: Uses PCA (Principal Component Analysis) to retain 75% of variance while reducing complexity.
- **Stacking Ensemble Model**: Combines multiple models (KNN, Random Forest, XGBoost, Gradient Boosting) for superior prediction accuracy.
- **Visualization**: Generates confusion matrices and classification reports for evaluation.

- **تنظيف البيانات**: معالجة القيم المفقودة والتكرارات.
- **معالجة البيانات**: تحويل البيانات النصية إلى أرقام وتقييس الميزات العددية.
- **موازنة البيانات (ADASYN)**: موازنة مجموعة البيانات لتحسين دقة النموذج للفئات الأقل تمثيلاً.
- **تقليل الأبعاد (PCA)**: استخدام تحليل المكونات الرئيسية للحفاظ على 75% من التباين مع تقليل التعقيد.
- **النموذج المجمع (Stacking Ensemble)**: دمج عدة نماذج (KNN, Random Forest, XGBoost, Gradient Boosting) لتحقيق أفضل دقة توقع.
- **التمثيل البياني**: إنشاء مصفوفة الارتباك وتقارير التصنيف للتقييم.

## Technologies Used | التقنيات المستخدمة
- Python
- Pandas & NumPy (Data Manipulation)
- Scikit-learn (Machine Learning & Preprocessing)
- Imbalanced-learn (ADASYN for balancing)
- XGBoost (Extreme Gradient Boosting)
- Matplotlib & Seaborn (Data Visualization)

## How to Run | كيفية التشغيل
1. Install requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```
2. Run the main script:
   ```bash
   python python.py
   ```

## Dataset | مجموعة البيانات
The project uses `survey lung cancer 1.csv` which contains various columns such as:
- Gender, Age, Smoking, Yellow Fingers, Anxiety, Peer Pressure, Chronic Disease, Fatigue, Allergy, Wheezing, Alcohol Consuming, Coughing, Shortness of Breath, Swallowing Difficulty, Chest Pain, and the target variable LUNG_CANCER.

## Model Evaluation | تقييم النموذج
The model achieves high accuracy by combining multiple classifiers through a Stacking Classifier, finalized with a Random Forest meta-learner.

يحقق النموذج دقة عالية من خلال دمج المصنفات المتعددة عبر Stacking Classifier، مع استخدام Random Forest كمصنف نهائي.
