import pandas as pd
import os

# Check your exact folder first
folder = r'C:\Users\siva2\OneDrive\Desktop\codecure'
print("📂 Files in folder:")
for f in os.listdir(folder):
    print(f"  {f}")

# Try common CSV names
csv_files = ['tox21.csv', 'tox21.xlsx', 'data.csv']
for filename in csv_files:
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        print(f"\n✅ FOUND: {filename}")
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        print("Shape:", df.shape)
        print("First 5 columns:", df.columns[:5].tolist())
        print("\nFirst 3 rows:")
        print(df.head(3))
        break
else:
    print("❌ No dataset found. Check filename!")
