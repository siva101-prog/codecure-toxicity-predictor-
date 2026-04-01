import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

print("🚀 Training Drug Toxicity Predictor...")

# Load data
df = pd.read_csv(r'C:\Users\siva2\OneDrive\Desktop\codecure\tox21.csv')

# Use first toxicity column as target (NR-AR), rest as features
target_col = 'NR-AR'
feature_cols = [col for col in df.columns if col != target_col and col != 'mol_id' and col != 'smiles']

X = df[feature_cols].fillna(0)
y = (df[target_col] > 0).astype(int)  # Binary: 0=Safe, 1=Toxic

print(f"Features: {len(feature_cols)}, Target: {target_col}")
print(f"Training on {len(y[y==1])} toxic compounds, {len(y[y==0])} safe")

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Results
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"\n✅ MODEL READY! Test Accuracy: {accuracy:.2%}")

# Feature importance (top 10)
importances = pd.DataFrame({
    'feature': feature_cols, 
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print("\nTop 5 Toxicity Predictors:")
print(importances.head())

# SAVE for demo
joblib.dump(model, 'toxicity_model.pkl')
joblib.dump(feature_cols, 'features.pkl')
importances.to_csv('feature_importance.csv', index=False)
X_test.head(10).to_csv('sample_data.csv', index=False)

print("\n💾 Files saved: model.pkl, features.pkl, feature_importance.csv")
print("🎉 PROTOTYPE COMPLETE! Run 'streamlit run app.py' next!")
