import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
from datetime import datetime
from model_utils import ModelManager
from PIL import Image, ImageTk

class LungCancerPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("التنبؤ بسرطان الرئة باستخدام التعلم الآلي - إعداد الطالب: علي لفته جبر")
        self.root.state('zoomed') # يفتح التطبيق مكبرًا على ويندوز
        self.root.configure(bg="#0F172A") 

        # الألوان والأنماط
        self.bg_color = "#0F172A"
        self.sidebar_color = "#1E293B"
        self.card_color = "#1E293B"
        self.input_bg = "#F8FAFC" # خلفية فاتحة للمدخلات لرؤية أفضل
        self.input_fg = "#0F172A" # نص داكن للمدخلات
        self.accent_color = "#38BDF8" 
        self.success_color = "#10B981" 
        self.danger_color = "#EF4444" 
        self.text_primary = "#F8FAFC"
        self.text_secondary = "#94A3B8"

        self.manager = ModelManager()
        self.patient_id_var = tk.StringVar(value=f"ID-{datetime.now().strftime('%M%S')}")
        self.patient_name_var = tk.StringVar(value="")
        
        self.features = [
            ("Gender (الجنس)", "GENDER", ["M", "F"]),
            ("Age (العمر)", "AGE", "int"),
            ("Smoking (التدخين)", "SMOKING", ["YES", "NO"]),
            ("Yellow Fingers (اصفرار الأصابع)", "YELLOW_FINGERS", ["YES", "NO"]),
            ("Anxiety (القلق)", "ANXIETY", ["YES", "NO"]),
            ("Peer Pressure (ضغط الأقران)", "PEER_PRESSURE", ["YES", "NO"]),
            ("Chronic Disease (أمراض مزمنة)", "CHRONIC DISEASE", ["YES", "NO"]),
            ("Fatigue (التعب)", "FATIGUE ", ["YES", "NO"]),
            ("Allergy (الحساسية)", "ALLERGY ", ["YES", "NO"]),
            ("Wheezing (الصفير)", "WHEEZING", ["YES", "NO"]),
            ("Alcohol Consuming (تعاطي الكحول)", "ALCOHOL CONSUMING", ["YES", "NO"]),
            ("Coughing (السعال)", "COUGHING", ["YES", "NO"]),
            ("Shortness of Breath (ضيق التنفس)", "SHORTNESS OF BREATH", ["YES", "NO"]),
            ("Swallowing Difficulty (صعوبة البلع)", "SWALLOWING DIFFICULTY", ["YES", "NO"]),
            ("Chest Pain (ألم الصدر)", "CHEST PAIN", ["YES", "NO"])
        ]
        
        self.inputs = {}
        self.last_prediction = None
        self.current_predictions = {} # Store current patient results for all algos
        self.setup_styles()
        self.setup_ui()
        self.initialize_model()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # نمط قائمة الاختيار (Combobox) بخلفية فاتحة
        style.configure("TCombobox", fieldbackground=self.input_bg, background="#CBD5E1", foreground=self.input_fg)
        style.map("TCombobox", fieldbackground=[('readonly', self.input_bg)])
        
        # نمط حقل الإدخال (Entry)
        style.configure("TEntry", fieldbackground=self.input_bg, foreground=self.input_fg)

    def setup_ui(self):
        # الشريط الجانبي
        self.sidebar = tk.Frame(self.root, bg=self.sidebar_color, width=320)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # الهوية التجارية
        brand_frame = tk.Frame(self.sidebar, bg=self.sidebar_color)
        brand_frame.pack(pady=(40, 20), padx=25, fill=tk.X)
        tk.Label(brand_frame, text="اعداد الطالب: علي لفته جبر", font=("Segoe UI", 10, "bold"), bg=self.sidebar_color, fg=self.text_primary).pack(anchor="w", pady=(5, 0))
        tk.Label(brand_frame, text="المشرف العلمي: د. زينة خليل", font=("Segoe UI", 10), bg=self.sidebar_color, fg=self.accent_color).pack(anchor="w", pady=(5, 0))

        # أزرار الشريط الجانبي (التنقل)
        self.create_nav_btn("🏠 Dashboard | لوحة التحكم", None, "top", active=True)
        self.reports_btn = self.create_nav_btn("📄 Medical Report | التقرير الطبي", self.show_medical_report, "top", state=tk.DISABLED)
        self.charts_btn = self.create_nav_btn("📊 Data Analytics | تحليلات البيانات", self.show_charts, "top", state=tk.DISABLED)
        
        # أزرار الخوارزميات (Algorithms Analysis)
        tk.Label(self.sidebar, text="ALGORITHM ANALYSIS | تحليل الخوارزميات", font=("Segoe UI", 8, "bold"), bg=self.sidebar_color, fg=self.accent_color, pady=10).pack(fill=tk.X)
        self.btn_knn = self.create_nav_btn("🔹 KNN Analysis", lambda: self.show_algo_analysis('knn'), "top", state=tk.DISABLED)
        self.btn_rf = self.create_nav_btn("🔹 RF Analysis", lambda: self.show_algo_analysis('rf'), "top", state=tk.DISABLED)
        self.btn_gb = self.create_nav_btn("🔹 GB Analysis", lambda: self.show_algo_analysis('gb'), "top", state=tk.DISABLED)
        self.btn_xgb = self.create_nav_btn("🔹 XGB Analysis", lambda: self.show_algo_analysis('xgb'), "top", state=tk.DISABLED)
        
        # مساحة العمل الرئيسية
        workspace = tk.Frame(self.root, bg=self.bg_color)
        workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # الرأس (Header)
        header = tk.Frame(workspace, bg=self.bg_color)
        header.pack(fill=tk.X, padx=40, pady=(40, 20))
        tk.Label(header, text="التنبؤ بسرطان الرئة باستخدام التعلم الآلي", font=("Segoe UI", 22, "bold"), bg=self.bg_color, fg=self.text_primary).pack(side=tk.LEFT)

        # منطقة المحتوى القابلة للتمرير
        content_container = tk.Frame(workspace, bg=self.bg_color)
        content_container.pack(fill=tk.BOTH, expand=True, padx=40)

        canvas = tk.Canvas(content_container, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=self.bg_color)

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # استخدام عرض ديناميكي للإطار الداخلي
        canvas_window = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        
        def configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. بطاقة هوية المريض
        id_card = tk.Frame(self.scroll_frame, bg=self.card_color, padx=30, pady=25)
        id_card.pack(fill=tk.X, pady=10)
        
        tk.Label(id_card, text="PATIENT IDENTIFICATION | هوية المريض", font=("Segoe UI", 10, "bold"), bg=self.card_color, fg=self.accent_color).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 20))
        
        tk.Label(id_card, text="FULL NAME | الاسم الكامل", bg=self.card_color, fg=self.text_secondary, font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w")
        name_ent = tk.Entry(id_card, textvariable=self.patient_name_var, bg=self.input_bg, fg=self.input_fg, insertbackground="black", relief=tk.FLAT, width=45, font=("Segoe UI", 12), bd=8)
        name_ent.grid(row=2, column=0, sticky="w", pady=(5, 0), padx=(0, 40))
        
        tk.Label(id_card, text="SYSTEM ID | المعرف", bg=self.card_color, fg=self.text_secondary, font=("Segoe UI", 10, "bold")).grid(row=1, column=1, sticky="w")
        id_ent = tk.Entry(id_card, textvariable=self.patient_id_var, bg=self.input_bg, fg=self.input_fg, insertbackground="black", relief=tk.FLAT, width=25, font=("Segoe UI", 12), bd=8)
        id_ent.grid(row=2, column=1, sticky="w", pady=(5, 0))

        # 2. شبكة أعراض المرض
        sym_card = tk.Frame(self.scroll_frame, bg=self.card_color, padx=30, pady=25)
        sym_card.pack(fill=tk.X, pady=10)
        
        tk.Label(sym_card, text="CLINICAL INDICATORS | المؤشرات السريرية", font=("Segoe UI", 10, "bold"), bg=self.card_color, fg=self.accent_color).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 20))

        for i, (label_text, key, options) in enumerate(self.features):
            row = (i // 2) + 1
            col_lbl = (i % 2) * 2
            col_ent = col_lbl + 1
            
            tk.Label(sym_card, text=label_text.upper(), bg=self.card_color, fg=self.text_secondary, font=("Segoe UI", 9, "bold")).grid(row=row, column=col_lbl, sticky="e", pady=15, padx=(20, 10))
            
            if isinstance(options, list):
                var = tk.StringVar(value=options[0])
                self.inputs[key] = var
                cb = ttk.Combobox(sym_card, textvariable=var, values=options, state="readonly", width=18, font=("Segoe UI", 11))
                cb.grid(row=row, column=col_ent, sticky="w", pady=15)
            else:
                ent = tk.Entry(sym_card, bg=self.input_bg, fg=self.input_fg, width=20, font=("Segoe UI", 11), bd=4, relief=tk.FLAT)
                ent.insert(0, "50")
                self.inputs[key] = ent
                ent.grid(row=row, column=col_ent, sticky="w", pady=15)

        # 3. زر التشغيل (معدل الحجم والموقع)
        self.main_predict_btn = tk.Button(self.scroll_frame, text="RUN AI DIAGNOSIS | بدء التشخيص", command=self.predict, 
                                        bg=self.accent_color, fg="#0F172A", font=("Segoe UI", 12, "bold"), 
                                        padx=40, pady=10, relief=tk.FLAT, cursor="hand2", 
                                        activebackground="#7DD3FC", activeforeground="#0F172A")
        self.main_predict_btn.pack(pady=(5, 20))

        # 4. مراقب النتائج
        self.monitor = tk.Frame(self.scroll_frame, bg=self.sidebar_color, padx=30, pady=30, highlightthickness=1)
        self.monitor.pack(fill=tk.X, pady=20)
        
        self.mon_status = tk.Label(self.monitor, text="SYSTEM STATUS: ONLINE", font=("Segoe UI", 9, "bold"), bg=self.sidebar_color, fg=self.text_secondary)
        self.mon_status.pack()
        
        self.mon_result = tk.Label(self.monitor, text="AWAITING INPUT", font=("Segoe UI", 26, "bold"), bg=self.sidebar_color, fg=self.accent_color)
        self.mon_result.pack(pady=10)
        
        self.mon_details = tk.Label(self.monitor, text="", font=("Segoe UI", 10), bg=self.sidebar_color, fg=self.text_secondary, justify="center")
        self.mon_details.pack(pady=5)

    def create_nav_btn(self, text, command, side, state=tk.NORMAL, active=False):
        btn = tk.Button(self.sidebar, text=text, command=command, state=state, 
                        bg=self.sidebar_color, fg=self.text_primary if active else self.text_secondary, 
                        font=("Segoe UI", 11), relief=tk.FLAT, activebackground="#334155",
                        padx=30, pady=18, anchor="w", cursor="hand2")
        btn.pack(side=side, fill=tk.X)
        return btn

    def initialize_model(self):
        if self.manager.load_model():
            self.mon_status.config(text="AI CORE: ACTIVE")
            self.charts_btn.config(state=tk.NORMAL)
            self.enable_algo_btns()
        else:
            self.mon_status.config(text="CALIBRATING...")
            self.root.after(100, self.train_model)

    def enable_algo_btns(self):
        for btn in [self.btn_knn, self.btn_rf, self.btn_gb, self.btn_xgb]:
            btn.config(state=tk.NORMAL)

    def train_model(self):
        try:
            self.manager.train_full_pipeline()
            self.manager.save_model()
            self.mon_status.config(text="CALIBRATION COMPLETE")
            self.charts_btn.config(state=tk.NORMAL)
            self.enable_algo_btns()
        except Exception as e:
            messagebox.showerror("Core Error", str(e))

    def predict(self):
        if not self.patient_name_var.get().strip():
            messagebox.showwarning("Incomplete Data", "Patient Name is mandatory for diagnostic logging.")
            return
        try:
            raw_data = {}
            for label, key, options in self.features:
                val = self.inputs[key].get()
                if key == "AGE": raw_data[key] = int(val)
                elif key == "GENDER": raw_data[key] = self.manager.label_encoders[key].transform([val])[0]
                else: raw_data[key] = 2 if val == "YES" else 1

            self.current_predictions = self.manager.predict(raw_data)
            predictions = self.current_predictions
            self.last_prediction = self.manager.label_encoders['LUNG_CANCER'].inverse_transform([predictions['stacking']])[0]
            
            color = self.danger_color if self.last_prediction == "YES" else self.success_color
            status = "ANALYSIS COMPLETE: CRITICAL RISK DETECTED" if self.last_prediction == "YES" else "ANALYSIS COMPLETE: NO ANOMALIES FOUND"
            status_ar = "إيجابي" if self.last_prediction == "YES" else "سلبي"
            
            p_name = self.patient_name_var.get().upper()
            self.mon_result.config(text=f"{p_name}\nFinal Result: {self.last_prediction} ({status_ar})", fg=color)
            self.mon_status.config(text=status, fg=color)
            
            # عرض نتائج بقية الخوارزميات مع دقتها
            details = []
            for algo in ['knn', 'rf', 'gb', 'xgb']:
                if algo in predictions:
                    res = self.manager.label_encoders['LUNG_CANCER'].inverse_transform([predictions[algo]])[0]
                    res_ar = "إيجابي" if res == "YES" else "سلبي"
                    acc_val = self.manager.accuracies.get(algo, 0)
                    details.append(f"{algo.upper()}: {res} ({res_ar})\nAccuracy: {acc_val:.2%}")
            
            self.mon_details.config(text="  |  ".join(details))
            self.monitor.config(highlightbackground=color)
            self.reports_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Logic Error", f"خطأ في المعالجة: {str(e)}")

    def show_charts(self):
        win = tk.Toplevel(self.root)
        win.title("AI ANALYTICS VISUALIZER")
        win.geometry("950x800")
        win.configure(bg=self.bg_color)
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        plots = [("CORRELATION", "plots/correlation_matrix.png"), ("EVALUATION", "plots/final_evaluation.png"), 
                 ("AGE TRENDS", "plots/age_distribution.png"), ("FEATURES", "plots/pca_component_importance.png")]
        
        self.tmp_imgs = []
        for name, path in plots:
            if os.path.exists(path):
                tab = tk.Frame(nb, bg=self.bg_color)
                nb.add(tab, text=name)
                with Image.open(path) as img_full:
                    img = img_full.resize((880, 650))
                    ph = ImageTk.PhotoImage(img)
                    self.tmp_imgs.append(ph)
                    tk.Label(tab, image=ph, bg=self.bg_color).pack(pady=10)

    def show_medical_report(self):
        if not self.last_prediction: return
        rep = tk.Toplevel(self.root)
        rep.title("DIAGNOSTIC ARCHIVE")
        rep.geometry("650x880")
        rep.configure(bg="#F8FAFC")
        
        header = tk.Frame(rep, bg=self.sidebar_color, pady=25)
        header.pack(fill=tk.X)
        tk.Label(header, text="OFFICIAL MEDICAL DIAGNOSIS", font=("Segoe UI", 16, "bold"), bg=self.sidebar_color, fg=self.accent_color).pack()
        
        content = tk.Text(rep, font=("Segoe UI", 10), padx=45, pady=40, relief=tk.FLAT, bg="white", fg="#0F172A")
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # تجهيز الحالة ثنائية اللغة
        status_en = "POSITIVE (RISK DETECTED)" if self.last_prediction == "YES" else "NEGATIVE (HEALTHY)"
        status_ar = "إيجابي (يوجد خطر)" if self.last_prediction == "YES" else "سلبي (سليم)"
        
        # بيانات متسقة للتقرير
        text = f"""
============================================================
              MEDICAL PREDICTION RECORD
              سجل التنبؤ الطبي الـذكـي
------------------------------------------------------------
مشروع: التنبؤ بسرطان الرئة باستخدام التعلم الآلي
اعداد الطالب: علي لفته جبر
المشرف العلمي: د. زينة خليل
============================================================
STAMP / التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SYSTEM LOG / السجل: Lung Cancer Prediction ML
------------------------------------------------------------

[ PATIENT DATA | بيانات المريض ]
- NAME / الاسم: {self.patient_name_var.get().upper()}
- UUID / المعرف: {self.patient_id_var.get().upper()}
- AGE / العمر: {self.inputs['AGE'].get()} Years
- GENDER / الجنس: {self.inputs['GENDER'].get()}

[ CLINICAL OBSERVATIONS | الملاحظات السريرية ]
- Chronic Disease / أمراض مزمنة: {self.inputs['CHRONIC DISEASE'].get()}
- Alcohol Intake / تعاطي الكحول: {self.inputs['ALCOHOL CONSUMING'].get()}
- Chest Pain / ألم الصدر: {self.inputs['CHEST PAIN'].get()}

[ EXAM RESULT | نتيجة الفحص ]
- FINAL RESULT / النتيجة النهائية: {status_en}
- ARABIC STATUS / الحالة بالعربية: {status_ar}

[ GUIDANCE | التوجيهات ]
{"PATIENT REQUIRES URGENT MEDICAL REVIEW" if self.last_prediction == "YES" else "NO PATHOLOGICAL INDICATIONS DETECTED"}
{"المريض يحتاج إلى مراجعة طبية عاجلة وفحوصات تكميلية" if self.last_prediction == "YES" else "لم يتم اكتشاف مؤشرات مرضية خطيرة حالياً"}

------------------------------------------------------------
VALIDATED BY: STACKING ENSEMBLE AI CORE
SECURITY HASH: {hex(id(self))}
============================================================
        """
        content.insert(tk.END, text)
        content.config(state=tk.DISABLED)
        
        tk.Button(rep, text="SAVE MEDICAL RECORD (.TXT)", bg="#0F172A", fg="white", 
                  font=("Segoe UI", 10, "bold"), pady=15, relief=tk.FLAT,
                  command=lambda: self.save_raw_report(text)).pack(fill=tk.X, padx=20, pady=(0, 20))

    def save_raw_report(self, text):
        f = filedialog.asksaveasfilename(defaultextension=".txt")
        if f:
            with open(f, 'w', encoding='utf-8') as file: file.write(text)
            messagebox.showinfo("Export", "Record Saved.")

    def show_algo_analysis(self, algo_key):
        report_text = self.manager.model_reports.get(algo_key, "No performance data available.")
        
        # Parse metrics from the text if possible for a prettier display
        metrics = {"Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", "F1-Score": "N/A", "ROC-AUC": "N/A"}
        lines = report_text.split('\n')
        for line in lines:
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                if key in metrics:
                    metrics[key] = val.strip()

        # Fallback to direct accuracy data if parsing failed (ensures no N/A is shown)
        if metrics["Accuracy"] == "N/A":
            acc_val = self.manager.accuracies.get(algo_key, 0)
            if acc_val > 0:
                metrics["Accuracy"] = f"{acc_val:.2%}"

        # Get current patient diagnosis
        current_res_text = "N/A (Run Diagnosis First)"
        if algo_key in self.current_predictions:
            res_val = self.current_predictions[algo_key]
            res_str = self.manager.label_encoders['LUNG_CANCER'].inverse_transform([res_val])[0]
            res_ar = "إيجابي (خطر)" if res_str == "YES" else "سلبي (سليم)"
            current_res_text = f"{res_str} ({res_ar})"
        
        algo_win = tk.Toplevel(self.root)
        algo_win.title(f"Performance Dashboard: {algo_key.upper()}")
        algo_win.geometry("800x520")
        algo_win.configure(bg="#0F172A")
        
        # Header area
        header = tk.Frame(algo_win, bg="#1E293B", pady=20)
        header.pack(fill=tk.X)
        tk.Label(header, text=f"ALGORITHM PERFORMANCE: {algo_key.upper()}", font=("Segoe UI", 16, "bold"), bg="#1E293B", fg=self.accent_color).pack()
        
        # Top Section: Current Patient Card
        patient_card = tk.Frame(algo_win, bg="#1E293B", padx=30, pady=20, highlightthickness=1, highlightbackground=self.accent_color)
        patient_card.pack(fill=tk.X, padx=30, pady=20)
        tk.Label(patient_card, text="CURRENT PATIENT DIAGNOSIS | تشخيص المريض الحالي", font=("Segoe UI", 9, "bold"), bg="#1E293B", fg=self.text_secondary).pack(anchor="w")
        tk.Label(patient_card, text=current_res_text, font=("Segoe UI", 22, "bold"), bg="#1E293B", fg="#F8FAFC").pack(anchor="w", pady=(5,0))

        # Middle Section: Metrics Dashboard (Grid of Cards)
        metrics_frame = tk.Frame(algo_win, bg="#0F172A")
        metrics_frame.pack(fill=tk.X, padx=30)
        
        # Metric categories to display as cards
        display_metrics = [
            ("ACCURACY", metrics["Accuracy"], "#38BDF8"),
            ("PRECISION", metrics["Precision"], "#10B981"),
            ("RECALL", metrics["Recall"], "#F59E0B"),
            ("F1-SCORE", metrics["F1-Score"], "#8B5CF6")
        ]

        for i, (m_name, m_val, m_color) in enumerate(display_metrics):
            card = tk.Frame(metrics_frame, bg="#1E293B", padx=15, pady=15, width=170, height=100)
            card.grid(row=0, column=i, padx=5, pady=10, sticky="nsew")
            card.grid_propagate(False)
            
            tk.Label(card, text=m_name, font=("Segoe UI", 8, "bold"), bg="#1E293B", fg=self.text_secondary).pack(anchor="center")
            tk.Label(card, text=m_val, font=("Segoe UI", 16, "bold"), bg="#1E293B", fg=m_color).pack(expand=True)
        
        metrics_frame.grid_columnconfigure((0,1,2,3), weight=1)

        # ROC-AUC Metric (Adding it centrally since it's important)
        auc_frame = tk.Frame(algo_win, bg="#0F172A", pady=10)
        auc_frame.pack(fill=tk.X, padx=30)
        tk.Label(auc_frame, text=f"ROC-AUC SCORE: {metrics['ROC-AUC']}", font=("Segoe UI", 12, "bold"), bg="#0F172A", fg="#F59E0B").pack()

        # Close Button
        tk.Button(algo_win, text="CLOSE DASHBOARD | إغلاق", bg="#1E293B", fg="white", font=("Segoe UI", 10, "bold"), 
                  command=algo_win.destroy, pady=12, relief=tk.FLAT, cursor="hand2", 
                  activebackground="#334155", activeforeground="white").pack(fill=tk.X, padx=30, pady=(10, 20))

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerPredictorApp(root)
    root.mainloop()
