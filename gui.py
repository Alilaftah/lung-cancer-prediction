import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier

class LungCancerPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Prediction System")
        self.root.geometry("900x750")
        self.root.configure(bg="#f4f7f6")

        # Define features labels and types - Now using YES/NO
        self.features = [
            ("Gender (الجنس)", "GENDER", ["M", "F"]),
            ("Age (العمر)", "AGE", "int"),
            ("Smoking (التدخين)", "SMOKING", ["YES", "NO"]),
            ("Yellow Fingers (اصفرار الأصابع)", "YELLOW_FINGERS", ["YES", "NO"]),
            ("Anxiety (القلق)", "ANXIETY", ["YES", "NO"]),
            ("Peer Pressure (ضغط الأقران)", "PEER_PRESSURE", ["YES", "NO"]),
            ("Chronic Disease (أمرض مزمنة)", "CHRONIC DISEASE", ["YES", "NO"]),
            ("Fatigue (التعب)", "FATIGUE ", ["YES", "NO"]),
            ("Allergy (الحساسية)", "ALLERGY ", ["YES", "NO"]),
            ("Wheezing (الصفير)", "WHEEZING", ["YES", "NO"]),
            ("Alcohol Consuming (الكحول)", "ALCOHOL CONSUMING", ["YES", "NO"]),
            ("Coughing (السعال)", "COUGHING", ["YES", "NO"]),
            ("Shortness of Breath (ضيق تنفس)", "SHORTNESS OF BREATH", ["YES", "NO"]),
            ("Swallowing Difficulty (صعوبة بلع)", "SWALLOWING DIFFICULTY", ["YES", "NO"]),
            ("Chest Pain (ألم الصدر)", "CHEST PAIN", ["YES", "NO"])
        ]
        
        self.inputs = {}
        self.model = None
        self.scaler = None
        self.pca = None
        self.le = {}

        self.setup_ui()
        self.train_on_background()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=80)
        header.pack(fill=tk.X)
        
        tk.Label(header, text="Lung Cancer Prediction Tool - أداة التنبؤ بسرطان الرئة", 
                 font=("Helvetica", 18, "bold"), bg="#2c3e50", fg="white").pack(pady=20)

        # Main Layout
        main_container = tk.Frame(self.root, bg="#f4f7f6")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Input Grid
        input_frame = tk.LabelFrame(main_container, text="Patient Information | بيانات المريض", 
                                   font=("Helvetica", 12, "bold"), bg="white", padx=20, pady=20)
        input_frame.pack(fill=tk.BOTH, expand=True)

        for i in range(4):
            input_frame.columnconfigure(i, weight=1)

        for i, (label_text, key, options) in enumerate(self.features):
            row = i // 2
            col_label = (i % 2) * 2
            col_entry = col_label + 1
            
            tk.Label(input_frame, text=label_text + ":", bg="white", font=("Helvetica", 10)).grid(row=row, column=col_label, sticky="e", pady=10, padx=5)
            
            if isinstance(options, list):
                var = tk.StringVar(value=options[0])
                self.inputs[key] = var
                cb = ttk.Combobox(input_frame, textvariable=var, values=options, state="readonly", width=15)
                cb.grid(row=row, column=col_entry, sticky="w", pady=10, padx=5)
            else:
                ent = ttk.Entry(input_frame, width=17)
                ent.insert(0, "50")
                self.inputs[key] = ent
                ent.grid(row=row, column=col_entry, sticky="w", pady=10, padx=5)

        # Predict Button
        self.predict_btn = tk.Button(main_container, text="Predict Result | عرض النتيجة", command=self.predict, 
                                   bg="#e74c3c", fg="white", font=("Helvetica", 14, "bold"), 
                                   padx=40, pady=10, relief=tk.FLAT)
        self.predict_btn.pack(pady=20)

        # Result Display
        self.result_label = tk.Label(main_container, text="Status: Model Training...", 
                                   font=("Helvetica", 16, "bold"), bg="#f4f7f6", fg="#2c3e50")
        self.result_label.pack()

    def train_on_background(self):
        try:
            file_path = 'survey lung cancer 1.csv'
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File {file_path} not found!")
                return

            df = pd.read_csv(file_path)
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            # Preprocessing
            self.le = {}
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.le[col] = le

            X = df.drop('LUNG_CANCER', axis=1)
            y = df['LUNG_CANCER']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            
            adasyn = ADASYN(random_state=88)
            X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_res)

            self.pca = PCA(n_components=0.75)
            X_train_pca = self.pca.fit_transform(X_train_scaled)

            base_models = [
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                ('gb', GradientBoostingClassifier(random_state=42))
            ]
            self.model = StackingClassifier(
                estimators=base_models,
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
            )
            self.model.fit(X_train_pca, y_train_res)
            self.result_label.config(text="Status: Ready for Prediction")
        except Exception as e:
            messagebox.showerror("Error", f"Training Failed: {str(e)}")

    def predict(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please wait for model to finish training.")
            return

        try:
            raw_data = {}
            for label, key, options in self.features:
                val = self.inputs[key].get()
                if key == "AGE":
                    raw_data[key] = int(val)
                elif key == "GENDER":
                    raw_data[key] = self.le[key].transform([val])[0]
                else:
                    # Conversion logic: Map YES to 2 and NO to 1
                    raw_data[key] = 2 if val == "YES" else 1

            input_df = pd.DataFrame([raw_data])
            input_scaled = self.scaler.transform(input_df)
            input_pca = self.pca.transform(input_scaled)
            
            prediction = self.model.predict(input_pca)[0]
            result_text = self.le['LUNG_CANCER'].inverse_transform([prediction])[0]
            
            color = "#e74c3c" if result_text == "YES" else "#27ae60"
            display_text = f"Result: {result_text} ({'High Risk' if result_text == 'YES' else 'Low Risk'})"
            self.result_label.config(text=display_text, fg=color)
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerPredictorApp(root)
    root.mainloop()
