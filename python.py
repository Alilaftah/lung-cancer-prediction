import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model_utils import ModelManager
import numpy as np
import os
import sys

# التأكد من ترميز UTF-8 لمخرجات وحدة التحكم (Console)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    # تهيئة مدير النموذج
    manager = ModelManager()
    
    print("--- 1) Starting Data Processing and Model Training ---")
    reports, X_test_pca, y_test = manager.train_full_pipeline()
    
    print("\n--- 2) Performance Reports ---")
    print(reports)
    
    manager.save_model()
    
    # --- 3) رسومات التقييم العلمي مع تعليقات توضيحية ---
    print("\n--- 3) Generating Annotated Scientific Evaluation Plots ---")
    
    # 3.1 مصفوفة الارتباك ومنحنى ROC (شكل مدمج)
    y_pred = manager.model.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(16, 7))
    
    # الشكل الفرعي أ: مصفوفة الارتباك (Confusion Matrix)
    plt.subplot(1, 2, 1)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', annot_kws={"size": 12})
    plt.title('Annotated Confusion Matrix Analysis', fontsize=13)
    plt.xlabel('Predicted Diagnosis', fontsize=11)
    plt.ylabel('Actual Diagnosis', fontsize=11)
    
    # الشكل الفرعي ب: منحنى ROC
    y_probs = manager.model.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='red', lw=3, label=f'Stacking Ensemble (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.1, color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13)
    plt.legend(loc="lower right")
    
    # تعليق علمي للشكل 4
    plt.figtext(0.5, -0.05, "Figure 4: Comparative evaluation of the predictive capability through Confusion Matrix (left) and ROC Curve (right) analysis.", 
                ha="center", fontsize=11, fontweight='bold', bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.savefig('plots/final_evaluation.png', bbox_inches='tight')
    
    # 3.3 أهمية مكونات PCA مع تعليق علمي (الشكل 5)
    rf_model = manager.model.named_estimators_['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 7))
    plt.title("Importance Ranking of PCA Components in Classification Decision", fontsize=15)
    bars = plt.bar(range(X_test_pca.shape[1]), importances[indices], align="center", color='teal')
    plt.xticks(range(X_test_pca.shape[1]), indices)
    plt.xlabel("Principle Component Index (Transformed Features)", fontsize=11)
    plt.ylabel("Importance Impact Score", fontsize=11)
    
    # تسمية الأعمدة
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=10)

    # تعليق علمي للشكل 5
    plt.figtext(0.5, -0.05, "Figure 5: Influence of identified principal components on the final classification decision of the Stacking model.", 
                ha="center", fontsize=11, fontweight='bold', bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.savefig('plots/pca_component_importance.png', bbox_inches='tight')
    
    print("\n--- Scientific annotated plots saved in 'plots' folder successfully ---")
    plt.show()

if __name__ == "__main__":
    main()