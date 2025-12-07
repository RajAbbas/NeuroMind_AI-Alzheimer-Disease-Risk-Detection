import zipfile
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- SETUP ---
output_dir = "Alzheimers_Results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. LOAD & CLEAN DATA (Using our robust logic)
print(">>> Loading TADPOLE Dataset...")
with zipfile.ZipFile('Tadpole_data.zip', 'r') as z:
    with z.open('TADPOLE_D1_D2.csv') as f:
        df = pd.read_csv(f, low_memory=False)

# Rename Columns (The Fix)
print(">>> Renaming Columns...")
search_map = {
    'ABETA': 'ABETA_UPENNBIOMK9', 
    'TAU': 'TAU_UPENNBIOMK9', 
    'PTAU': 'PTAU_UPENNBIOMK9'
}
rename_dict = {}
for std, key in search_map.items():
    matches = [c for c in df.columns if key in c]
    if matches: rename_dict[matches[0]] = std
df = df.rename(columns=rename_dict)

# Select Features (Proteins + MRI + Genetics + Demographics)
features = ['ABETA', 'TAU', 'PTAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 
            'Entorhinal', 'Fusiform', 'AGE', 'PTGENDER', 'APOE4', 'DX_bl']

# Filter only rows that have these columns
missing_cols = [c for c in features if c not in df.columns]
if missing_cols:
    print(f"❌ Error: Missing columns {missing_cols}")
    exit()

df = df[features].copy()

# Clean "Messy" Numbers (Remove < and >)
print(">>> Cleaning Numerical Values...")
for col in ['ABETA', 'TAU', 'PTAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 'Entorhinal', 'Fusiform']:
    df[col] = df[col].astype(str).str.replace(r'[><]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing values (We only want high-quality data for this model)
df = df.dropna()
print(f"Final Training Set: {df.shape[0]} patients")

# 2. FEATURE ENGINEERING (The Innovation)
print(">>> Engineering 'Neurotoxic Ratio'...")
# Ratio: High PTAU (Bad) / Low ABETA (Bad) = Very High Score
df['Neurotoxic_Ratio'] = df['PTAU'] / df['ABETA']

# Brain Reserve: Hippocampus size relative to head size
df['Hippocampus_Fraction'] = df['Hippocampus'] / df['WholeBrain']

# 3. PREPARE TARGET
# DX_bl Codes: CN (Cognitively Normal), SMC (Significant Memory Concern), EMCI/LMCI (Mild Impairment), AD (Alzheimer's)
# Binary Task: 0 = Healthy (CN, SMC), 1 = Sick (MCI, AD)
target_map = {'CN': 0, 'SMC': 0, 'EMCI': 1, 'LMCI': 1, 'AD': 1}
df['Target'] = df['DX_bl'].map(target_map)
df = df.dropna(subset=['Target']) # Drop unknown diagnoses

# Encode Gender
df['PTGENDER'] = LabelEncoder().fit_transform(df['PTGENDER'])

# 4. TRAIN MODEL
X = df.drop(['DX_bl', 'Target'], axis=1)
y = df['Target']

# Stratify is important because we have imbalanced classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n>>> Training XGBoost on {X_train.shape[0]} samples...")
model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 5. EVALUATE
print("\n>>> Evaluating Performance...")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
roc = roc_auc_score(y_test, probs)

print(f"🚀 Test Accuracy: {acc:.2%}")
print(f"📊 ROC-AUC Score: {roc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, preds))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (Acc: {acc:.1%})")
plt.xlabel("Predicted (0=Healthy, 1=MCI/AD)")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

# 6. EXPLAINABILITY (SHAP)
print("\n>>> Generating SHAP Explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Importance: Does the 'Ratio' matter?", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/shap_summary_alz.png")
plt.close()

print(f"Saved charts to '{output_dir}'. Check 'shap_summary_alz.png'!")